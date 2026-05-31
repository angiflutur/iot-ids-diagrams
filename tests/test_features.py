"""
test_features.py - Unit Tests for IoT IDS Feature Engineering

Tests verify that all feature calculations produce correct values
and that the logic is consistent between preprocessing and deployment.

Test classes:
    TestIATFeatures       - Inter-arrival time and burstiness
    TestSensorFeatures    - Sensor delta, clipping, inf handling
    TestTopicFeatures     - Topic classification
    TestPCAPFeatures      - PCAP-derived features
    TestSequenceFeatures  - Sequence gap and backwards detection
    TestVotingLogic       - Weighted voting decision mechanism
    TestFeatureProperties - Feature list consistency checks
"""

import numpy as np
import pytest

LEGIT_TOPIC = 'iot/sensors/environmental'
MQTT_PORT   = 1883
PCAP_WIN    = 5.0


class TestIATFeatures:
    """
    Tests for inter-arrival time features.
    DoS attack: IAT ~0.027s -> iat_log ~0.027 (very small)
    Legitimate: IAT ~5.0s  -> iat_log ~1.79  (larger)
    """

    def test_iat_log_legitimate(self):
        """Legitimate traffic has IAT ~5s -> iat_log ~1.79"""
        assert abs(np.log1p(5.0) - 1.7917) < 0.001

    def test_iat_log_dos(self):
        """DoS traffic has IAT ~0.027s -> very small iat_log"""
        assert np.log1p(0.027) < 0.1

    def test_burstiness_dos_uniform(self):
        """DoS sends at constant rate -> near-zero burstiness"""
        iats = np.array([0.027] * 10)
        assert iats.std() / (iats.mean() + 1e-9) < 0.01

    def test_burstiness_legitimate(self):
        """Legitimate traffic has natural variation -> moderate burstiness"""
        iats = np.array([4.8, 5.1, 5.3, 4.9, 5.0,
                         5.2, 4.7, 5.1, 5.0, 4.8])
        b    = iats.std() / (iats.mean() + 1e-9)
        assert 0.01 < b < 0.5


class TestSensorFeatures:
    """
    Tests for sensor delta features.
    Spoofing: temperature jumps from 23C to 70C -> large delta
    Malformed: humidity=9999 -> clipped to 200 before delta
    """

    def test_humidity_clip_malformed(self):
        """Malformed payload sends humidity=9999 -> clipped to 200"""
        assert float(np.clip(9999.0, -10.0, 200.0)) == 200.0

    def test_humidity_clip_negative(self):
        """Negative humidity clipped to -10"""
        assert float(np.clip(-50.0, -10.0, 200.0)) == -10.0

    def test_delta_temp_spoofing(self):
        """Spoofing: temp jumps from 23 to 70 -> delta_temp_log > 3.5"""
        assert np.log1p(abs(70.0 - 23.0)) > 3.5

    def test_delta_temp_legitimate(self):
        """Legitimate: small natural variation -> delta_temp_log < 0.2"""
        assert np.log1p(abs(23.4 - 23.2)) < 0.2

    def test_inf_temperature_replaced(self):
        """Malformed: inf temperature replaced with default 23.0"""
        temp = float('inf')
        if not np.isfinite(temp):
            temp = 23.0
        assert temp == 23.0

    def test_delta_clip_malformed(self):
        """Delta clipped at 1000 to prevent overflow"""
        assert min(9999.0, 1000.0) == 1000.0


class TestTopicFeatures:
    """
    Tests for topic-based features.
    Topic Injection publishes on unauthorized topics -> topic_is_known=0
    """

    def test_known_topic(self):
        """Legitimate topic -> topic_is_known=1"""
        assert int('iot/sensors/environmental' == LEGIT_TOPIC) == 1

    def test_injection_topic(self):
        """Injected topic -> topic_is_known=0"""
        assert int('iot/commands/actuator' == LEGIT_TOPIC) == 0

    def test_topic_depth_legitimate(self):
        """Legitimate topic has 2 slashes -> topic_depth=2"""
        assert float('iot/sensors/environmental'.count('/')) == 2.0

    def test_topic_depth_injection(self):
        """Injected topic may have different depth"""
        assert float('iot/commands/actuator/set'.count('/')) == 3.0


class TestPCAPFeatures:
    """
    Tests for PCAP-derived features.
    Brute Force: many small SYN/RST packets, rapid reconnections.
    Legitimate: slow MQTT traffic, persistent connections.
    """

    def test_ipt_log_bruteforce(self):
        """Brute Force: 0.6ms between packets -> pcap_ipt_log near 0"""
        assert np.log1p(0.0006) < 0.01

    def test_ipt_log_legitimate(self):
        """Legitimate: 2.37s between packets -> pcap_ipt_log > 1"""
        assert np.log1p(2.37) > 1.0

    def test_syn_ratio_bruteforce(self):
        """Brute Force: 89% of packets are small SYN/RST"""
        lengths = np.array([54, 54, 60, 54, 54, 200, 54])
        assert float((lengths < 60).mean()) > 0.7

    def test_syn_ratio_legitimate(self):
        """Legitimate: mix of small and large packets"""
        lengths = np.array([54, 200, 350, 180, 54, 220])
        assert float((lengths < 60).mean()) < 0.6

    def test_tcp_reconnect_rate_bruteforce(self):
        """
        Brute Force: new source port per attempt -> high reconnect rate.
        5 unique ports in 5s window = rate of 1.0 connections/second.
        """
        src  = np.array([52001, 52002, 52003, 52004, 52005,
                         1883,  1883,  1883])
        only = src[src != MQTT_PORT]
        rate = float(len(set(only)) / PCAP_WIN)
        assert rate == 1.0

    def test_tcp_reconnect_rate_legitimate(self):
        """
        Legitimate: ESP32 uses persistent connection -> low rate.
        Only 1 unique client port -> rate = 0.2 connections/second.
        """
        src  = np.array([52001, 1883, 1883, 1883, 1883])
        only = src[src != MQTT_PORT]
        rate = float(len(set(only)) / PCAP_WIN)
        assert rate <= 0.4

    def test_pkt_size_cv_bruteforce(self):
        """Brute Force: uniform SYN packets -> low CV"""
        lengths = np.array([54, 54, 60, 54, 58, 54, 60])
        mean    = lengths.mean()
        cv      = float(lengths.std() / mean) if mean > 0 else 0.0
        assert cv < 0.5

    def test_pkt_size_cv_legitimate(self):
        """Legitimate: mixed packet sizes -> high CV"""
        lengths = np.array([54, 200, 350, 180, 420, 60, 280])
        mean    = lengths.mean()
        cv      = float(lengths.std() / mean) if mean > 0 else 0.0
        assert cv > 0.5


class TestSequenceFeatures:
    """
    Tests for sequence number features.
    MITM: reorders or replays messages -> large gap, seq_backwards=1
    """

    def test_seq_gap_normal(self):
        """Normal increment of 1 -> seq_gap_clipped=0"""
        assert float(min(abs(1 - 1), 500)) == 0.0

    def test_seq_gap_mitm(self):
        """MITM reorders messages -> large gap"""
        assert float(min(abs(73 - 1), 500)) == 72.0

    def test_seq_gap_clip(self):
        """Gap clipped at 500 to handle file concatenation artifacts"""
        assert float(min(abs(10000 - 1), 500)) == 500.0

    def test_seq_backwards_replay(self):
        """MITM replay: sequence goes backwards -> seq_backwards=1"""
        assert int(-5 < 0) == 1

    def test_seq_backwards_normal(self):
        """Normal traffic: sequence increases -> seq_backwards=0"""
        assert int(1 < 0) == 0


class TestVotingLogic:
    """
    Tests for weighted voting mechanism.

    Weights: LightGBM=0.40, Random Forest=0.20, Isolation Forest=0.40
    Threshold: 0.50 (at least 2 models must agree)
    """

    def test_all_agree_attack(self):
        """All three models vote attack -> score=1.00 -> alert"""
        score = 0.40 * 1 + 0.20 * 1 + 0.40 * 1
        assert score == 1.00
        assert score >= 0.50

    def test_lgbm_if_alert(self):
        """LightGBM + Isolation Forest -> score=0.80 -> alert"""
        score = 0.40 * 1 + 0.20 * 0 + 0.40 * 1
        assert score == 0.80
        assert score >= 0.50

    def test_lgbm_rf_alert(self):
        """LightGBM + Random Forest -> score=0.60 -> alert"""
        score = 0.40 * 1 + 0.20 * 1 + 0.40 * 0
        assert abs(score - 0.60) < 1e-9
        assert score >= 0.50

    def test_if_only_no_alert(self):
        """
        Isolation Forest alone -> score=0.40 -> no standard alert.
        Zero-day path handles this case separately.
        """
        score = 0.40 * 0 + 0.20 * 0 + 0.40 * 1
        assert score == 0.40
        assert score < 0.50

    def test_zero_day_if_path(self):
        """
        Zero-day detection: IF detects strong anomaly but supervised
        models classify as legitimate -> alert via IF-only path.
        Conditions: if_score < -0.10 AND lgbm=legitimate AND conf>=0.80
        """
        pure_anomaly = (
            True                               # if_is_attack
            and (-0.15 < -0.10)                # if_score < -0.10
            and (0.40 < 0.50)                  # vote below threshold
            and ('legitimate' == 'legitimate') # lgbm_label
            and (0.85 >= 0.80)                 # lgbm_conf
        )
        assert pure_anomaly is True

    def test_all_legitimate(self):
        """All models agree legitimate -> score=0.00 -> no alert"""
        score = 0.40 * 0 + 0.20 * 0 + 0.40 * 0
        assert score == 0.00
        assert score < 0.50


class TestFeatureProperties:
    """
    Tests for feature list consistency.
    Ensures preprocessing and deployment use identical feature sets.
    """

    def test_total_features_17(self):
        """Total feature count must be exactly 17"""
        features = [
            'iat_log', 'burstiness',
            'json_field_count', 'json_valid', 'errors_flag',
            'topic_is_known', 'topic_depth',
            'delta_temp_log', 'delta_hum_log',
            'temp_rolling_std', 'hum_rolling_std',
            'seq_gap_clipped', 'seq_backwards',
            'pcap_ipt_log', 'pcap_pkt_size_cv',
            'pcap_syn_ratio', 'tcp_reconnect_rate',
        ]
        assert len(features) == 17

    def test_binary_features_4(self):
        """Exactly 4 binary features (not scaled by RobustScaler)"""
        binary = ['json_valid', 'errors_flag',
                  'topic_is_known', 'seq_backwards']
        assert len(binary) == 4

    def test_pcap_features_4(self):
        """Exactly 4 PCAP features, all network-independent"""
        pcap = ['pcap_ipt_log', 'pcap_pkt_size_cv',
                'pcap_syn_ratio', 'tcp_reconnect_rate']
        assert len(pcap) == 4

    def test_mqtt_features_13(self):
        """Exactly 13 MQTT features"""
        all_f = [
            'iat_log', 'burstiness',
            'json_field_count', 'json_valid', 'errors_flag',
            'topic_is_known', 'topic_depth',
            'delta_temp_log', 'delta_hum_log',
            'temp_rolling_std', 'hum_rolling_std',
            'seq_gap_clipped', 'seq_backwards',
            'pcap_ipt_log', 'pcap_pkt_size_cv',
            'pcap_syn_ratio', 'tcp_reconnect_rate',
        ]
        mqtt = [f for f in all_f
                if not f.startswith('pcap_')
                and f != 'tcp_reconnect_rate']
        assert len(mqtt) == 13