"""
feature_state.py - MQTT Feature Engineering
Reproduces exactly the preprocessing pipeline from training.
Maintains sliding windows for behavioral feature computation.

17 features total:
    Timing (2):       iat_log, burstiness
    Protocol (3):     json_field_count, json_valid, errors_flag
    Topic (2):        topic_is_known, topic_depth
    Sensor Delta (4): delta_temp_log, delta_hum_log,
                      temp_rolling_std, hum_rolling_std
    Sequence (2):     seq_gap_clipped, seq_backwards
    PCAP (4):         pcap_ipt_log, pcap_pkt_size_cv,
                      pcap_syn_ratio, tcp_reconnect_rate
"""

import collections
import numpy as np

LEGIT_TOPIC = 'iot/sensors/environmental'
IAT_WIN     = 10
TEMP_WIN    = 5


class FeatureState:
    """
    Stateful feature extractor for MQTT messages.
    Must be instantiated once per subscriber and reused
    across messages to maintain sliding window state.
    """

    def __init__(self):
        self.iat_window    = collections.deque(maxlen=IAT_WIN)
        self.temp_window   = collections.deque(maxlen=TEMP_WIN)
        self.hum_window    = collections.deque(maxlen=TEMP_WIN)
        self.prev_temp     = None
        self.prev_hum      = None
        self.prev_sequence = None
        self.prev_ts       = None

    def compute(self, recv_time, topic, data):
        """
        Compute all 13 MQTT features from a received message.

        Args:
            recv_time : float  - Unix timestamp of reception
            topic     : str    - MQTT topic string
            data      : dict   - Parsed JSON payload

        Returns:
            dict with 13 MQTT features + _temperature, _humidity
            (private fields used for alert payload, not ML input)
        """

        # Timing: IAT and burstiness
        iat = max(0.0, recv_time - self.prev_ts) \
              if self.prev_ts is not None else 5.0
        self.prev_ts = recv_time
        self.iat_window.append(iat)

        iat_log    = float(np.log1p(iat))
        burstiness = 0.0
        if len(self.iat_window) >= 2:
            arr        = np.array(self.iat_window)
            burstiness = float(arr.std() / (arr.mean() + 1e-9))

        # Protocol features
        json_valid       = int(isinstance(data, dict) and len(data) > 0)
        json_field_count = float(len(data) if isinstance(data, dict) else 0)
        errors           = float(data.get('errors', 0)) \
                           if isinstance(data, dict) else 0.0
        errors_flag      = int(errors > 0)

        # Topic features
        topic_is_known = int(topic == LEGIT_TOPIC)
        topic_depth    = float(topic.count('/'))

        # Sensor values with sanitization
        sensor_data = data.get('data', {}) if isinstance(data, dict) else {}
        if not isinstance(sensor_data, dict):
            sensor_data = {}

        try:
            temperature = float(sensor_data.get('temperature', 23.0))
            temperature = 23.0 if not np.isfinite(temperature) else temperature
        except Exception:
            temperature = 23.0

        try:
            humidity = float(sensor_data.get('humidity', 40.0))
            humidity = 40.0 if not np.isfinite(humidity) else humidity
        except Exception:
            humidity = 40.0

        # Clip humidity: malformed payloads can send 9999%
        humidity_clipped = float(np.clip(humidity, -10.0, 200.0))

        # Sensor delta features
        delta_temp = min(abs(temperature - self.prev_temp), 1000.0) \
                     if self.prev_temp is not None else 0.0
        delta_hum  = min(abs(humidity_clipped - self.prev_hum), 500.0) \
                     if self.prev_hum is not None else 0.0

        self.prev_temp = temperature
        self.prev_hum  = humidity_clipped

        delta_temp_log = float(np.log1p(delta_temp))
        delta_hum_log  = float(np.log1p(delta_hum))

        # Rolling standard deviation over last 5 messages
        self.temp_window.append(temperature)
        self.hum_window.append(humidity_clipped)

        temp_rolling_std = 0.0
        hum_rolling_std  = 0.0
        if len(self.temp_window) >= 2:
            temp_arr = np.array(list(self.temp_window), dtype=float)
            hum_arr  = np.array(list(self.hum_window),  dtype=float)
            temp_rolling_std = float(np.std(temp_arr))
            hum_rolling_std  = float(np.std(hum_arr))

        # Sequence features
        sequence = data.get('sequence', None) \
                   if isinstance(data, dict) else None
        seq_gap_clipped = 0.0
        seq_backwards   = 0
        if sequence is not None and self.prev_sequence is not None:
            try:
                seq_diff        = int(sequence) - int(self.prev_sequence)
                seq_gap_clipped = float(min(abs(seq_diff - 1), 500))
                seq_backwards   = int(seq_diff < 0)
            except Exception:
                pass
        self.prev_sequence = sequence

        return {
            'iat_log':           iat_log,
            'burstiness':        burstiness,
            'json_field_count':  json_field_count,
            'json_valid':        float(json_valid),
            'errors_flag':       float(errors_flag),
            'topic_is_known':    float(topic_is_known),
            'topic_depth':       topic_depth,
            'delta_temp_log':    delta_temp_log,
            'delta_hum_log':     delta_hum_log,
            'temp_rolling_std':  temp_rolling_std,
            'hum_rolling_std':   hum_rolling_std,
            'seq_gap_clipped':   seq_gap_clipped,
            'seq_backwards':     float(seq_backwards),
            '_temperature':      temperature,
            '_humidity':         humidity,
        }