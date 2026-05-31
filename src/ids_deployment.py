"""
ids_deployment.py - Hybrid IDS for Raspberry Pi
Real-time intrusion detection for MQTT-based IoT environments.

Architecture:
    Thread 1: PCAP Capture (pyshark, tcp port 1883)
    Thread 2: MQTT Subscriber (paho-mqtt)
    Thread 3: IDS Engine (LightGBM + Random Forest + Isolation Forest)

Weighted voting: 0.40 * LGBM + 0.20 * RF + 0.40 * IF >= 0.50 -> ALERT
17 network-independent features: 13 MQTT + 4 PCAP
"""

import threading
import time
import json
import queue
import numpy as np
import pandas as pd
import joblib
import collections

# ── Config ────────────────────────────────────────────────────────────────────

BROKER_IP      = 'localhost'
BROKER_PORT    = 1883
BROKER_USER    = 'iot-user'
BROKER_PASS    = 'shadow200'
NETWORK_IFACE  = 'any'
MQTT_PORT      = 1883
PCAP_WIN_SEC   = 5.0
LEGIT_TOPIC    = 'iot/sensors/environmental'
ALERT_TOPIC    = 'ids/alerts'
MODELS_BASE    = '/home/angelica/Desktop/ids_27/models'

W_LGBM         = 0.40
W_RF           = 0.20
W_IFOREST      = 0.40
VOTE_THRESHOLD = 0.50
MIN_CONFIDENCE = 0.50

# ── Shared state ──────────────────────────────────────────────────────────────

pcap_buffer = collections.deque(maxlen=10000)
pcap_lock   = threading.Lock()
mqtt_queue  = queue.Queue(maxsize=1000)

# ── Feature engineering ───────────────────────────────────────────────────────

def get_pcap_features(window_start, window_end):
    """
    Extract 4 PCAP features from packets in [window_start, window_end].
    All features are network-independent (no absolute IP/port values).

    Returns:
        pcap_ipt_log      : log(1 + median inter-packet time)
        pcap_pkt_size_cv  : coefficient of variation of packet sizes
        pcap_syn_ratio    : proportion of packets < 60 bytes (SYN/RST)
        tcp_reconnect_rate: unique source ports != 1883, per second
                           (novel feature for Brute Force detection)
    """
    with pcap_lock:
        pkts = [p for p in pcap_buffer
                if window_start <= p['timestamp'] <= window_end]

    if not pkts:
        return {
            'pcap_ipt_log':       0.0,
            'pcap_pkt_size_cv':   0.0,
            'pcap_syn_ratio':     0.0,
            'tcp_reconnect_rate': 0.0,
        }

    lengths    = np.array([p['length']   for p in pkts], dtype=float)
    timestamps = np.array([p['timestamp']for p in pkts], dtype=float)
    src_ports  = np.array([p['src_port'] for p in pkts], dtype=int)

    ipt_median = float(np.median(np.diff(np.sort(timestamps)))) \
                 if len(timestamps) > 1 else 0.0

    len_mean         = lengths.mean()
    pcap_pkt_size_cv = float(lengths.std() / len_mean) \
                       if len_mean > 0 else 0.0

    src_only           = src_ports[src_ports != MQTT_PORT]
    tcp_reconnect_rate = float(len(set(src_only)) / PCAP_WIN_SEC)

    return {
        'pcap_ipt_log':       float(np.log1p(ipt_median)),
        'pcap_pkt_size_cv':   pcap_pkt_size_cv,
        'pcap_syn_ratio':     float((lengths < 60).mean()),
        'tcp_reconnect_rate': tcp_reconnect_rate,
    }


def weighted_vote(lgbm_label, rf_is_attack, if_is_attack):
    """
    Weighted voting: requires at least 2 models to agree.
    No single model can trigger an alert alone (max score = 0.40).
    """
    score = (
        W_LGBM    * (1.0 if lgbm_label != 'legitimate' else 0.0) +
        W_RF      * (1.0 if rf_is_attack else 0.0) +
        W_IFOREST * (1.0 if if_is_attack else 0.0)
    )
    return score >= VOTE_THRESHOLD, score


def predict(lgbm, rf, iforest, scaler,
            features_all, features_used,
            binary_cols, numeric_cols,
            id2label, row_dict):
    """Apply all three models and return predictions."""
    clean = {
        feat: (float(v) if np.isfinite(float(v)) else 0.0)
        for feat in features_all
        for v in [row_dict.get(feat, 0.0)]
    }

    df          = pd.DataFrame([clean], columns=features_all)
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    df_used     = df[features_used]

    lgbm_enc    = int(lgbm.predict(df_used)[0])
    lgbm_proba  = lgbm.predict_proba(df_used)[0]
    lgbm_label  = id2label[lgbm_enc]
    lgbm_conf   = float(lgbm_proba[lgbm_enc])

    rf_pred      = int(rf.predict(df_used)[0])
    rf_is_attack = bool(rf_pred)
    rf_conf      = float(rf.predict_proba(df_used)[0][rf_pred])

    if_pred      = int(iforest.predict(df_used)[0])
    if_is_attack = (if_pred == -1)
    if_score     = float(iforest.decision_function(df_used)[0])

    is_attack, vote_score = weighted_vote(
        lgbm_label, rf_is_attack, if_is_attack
    )

    return (lgbm_label, lgbm_conf,
            rf_is_attack, rf_conf,
            if_is_attack, if_score,
            is_attack, vote_score)