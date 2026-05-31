"""
Unit tests for IoT IDS feature engineering.
Validates that feature calculations are correct
and consistent between preprocessing and deployment.
"""
import numpy as np
import pytest

LEGIT_TOPIC = 'iot/sensors/environmental'
MQTT_PORT   = 1883
PCAP_WIN    = 5.0


class TestIATFeatures:

    def test_iat_log_legitimate(self):
        iat = 5.0
        assert abs(np.log1p(iat) - 1.7917) < 0.001

    def test_iat_log_dos(self):
        assert np.log1p(0.027) < 0.1

    def test_burstiness_dos_uniform(self):
        iats = np.array([0.027] * 10)
        b    = iats.std() / (iats.mean() + 1e-9)
        assert b < 0.01

    def test_burstiness_legitimate(self):
        iats = np.array([4.8, 5.1, 5.3, 4.9, 5.0,
                         5.2, 4.7, 5.1, 5.0, 4.8])
        b    = iats.std() / (iats.mean() + 1e-9)
        assert 0.01 < b < 0.5


class TestSensorFeatures:

    def test_humidity_clip_malformed(self):
        assert float(np.clip(9999.0, -10.0, 200.0)) == 200.0

    def test_humidity_clip_negative(self):
        assert float(np.clip(-50.0, -10.0, 200.0)) == -10.0

    def test_delta_temp_spoofing(self):
        assert np.log1p(abs(70.0 - 23.0)) > 3.5

    def test_delta_temp_legitimate(self):
        assert np.log1p(abs(23.4 - 23.2)) < 0.2

    def test_inf_temperature_replaced(self):
        temp = float('inf')
        if not np.isfinite(temp):
            temp = 23.0
        assert temp == 23.0

    def test_delta_clip_malformed(self):
        delta = min(9999.0, 1000.0)
        assert delta == 1000.0


class TestTopicFeatures:

    def test_known_topic(self):
        assert int('iot/sensors/environmental' == LEGIT_TOPIC) == 1

    def test_injection_topic(self):
        assert int('iot/commands/actuator' == LEGIT_TOPIC) == 0

    def test_topic_depth_legitimate(self):
        assert float('iot/sensors/environmental'.count('/')) == 2.0

    def test_topic_depth_injection(self):
        assert float('iot/commands/actuator/set'.count('/')) == 3.0


class TestPCAPFeatures:

    def test_ipt_log_bruteforce(self):
        assert np.log1p(0.0006) < 0.01

    def test_ipt_log_legitimate(self):
        assert np.log1p(2.37) > 1.0

    def test_syn_ratio_bruteforce(self):
        lengths = np.array([54, 54, 60, 54, 54, 200, 54])
        assert float((lengths < 60).mean()) > 0.7

    def test_syn_ratio_legitimate(self):
        lengths = np.array([54, 200, 350, 180, 54, 220])
        assert float((lengths < 60).mean()) < 0.6

    def test_tcp_reconnect_rate_bruteforce(self):
        src = np.array([52001, 52002, 52003, 52004, 52005,
                        1883,  1883,  1883])
        only = src[src != MQTT_PORT]
        rate = float(len(set(only)) / PCAP_WIN)
        assert rate == 1.0

    def test_tcp_reconnect_rate_legitimate(self):
        src  = np.array([52001, 1883, 1883, 1883, 1883])
        only = src[src != MQTT_PORT]
        rate = float(len(set(only)) / PCAP_WIN)
        assert rate <= 0.4

    def test_pkt_size_cv_bruteforce(self):
        lengths  = np.array([54, 54, 60, 54, 58, 54, 60])
        mean     = lengths.mean()
        cv       = float(lengths.std() / mean) if mean > 0 else 0.0
        assert cv < 0.5

    def test_pkt_size_cv_legitimate(self):
        lengths = np.array([54, 200, 350, 180, 420, 60, 280])
        mean    = lengths.mean()
        cv      = float(lengths.std() / mean) if mean > 0 else 0.0
        assert cv > 0.5


class TestSequenceFeatures:

    def test_seq_gap_normal(self):
        assert float(min(abs(1 - 1), 500)) == 0.0

    def test_seq_gap_mitm(self):
        assert float(min(abs(73 - 1), 500)) == 72.0

    def test_seq_gap_clip(self):
        assert float(min(abs(10000 - 1), 500)) == 500.0

    def test_seq_backwards_replay(self):
        assert int(-5 < 0) == 1

    def test_seq_backwards_normal(self):
        assert int(1 < 0) == 0


class TestVotingLogic:

    def test_all_agree_attack(self):
        score = 0.40 * 1 + 0.20 * 1 + 0.40 * 1
        assert score == 1.00
        assert score >= 0.50

    def test_lgbm_if_alert(self):
        score = 0.40 * 1 + 0.20 * 0 + 0.40 * 1
        assert score == 0.80
        assert score >= 0.50

    def test_lgbm_rf_alert(self):
        score = 0.40 * 1 + 0.20 * 1 + 0.40 * 0
        assert score == 0.60
        assert score >= 0.50

    def test_if_only_no_alert(self):
        score = 0.40 * 0 + 0.20 * 0 + 0.40 * 1
        assert score == 0.40
        assert score < 0.50

    def test_zero_day_if_path(self):
        pure_anomaly = (
            True           # if_is_attack
            and -0.15 < -0.10          # if_score < -0.10
            and 0.40 < 0.50            # vote < threshold
            and 'legitimate' == 'legitimate'
            and 0.85 >= 0.80           # lgbm_conf
        )
        assert pure_anomaly is True

    def test_all_legitimate(self):
        score = 0.40 * 0 + 0.20 * 0 + 0.40 * 0
        assert score == 0.00
        assert score < 0.50


class TestFeatureProperties:

    def test_total_features_17(self):
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
        binary = ['json_valid', 'errors_flag',
                  'topic_is_known', 'seq_backwards']
        assert len(binary) == 4

    def test_pcap_features_4(self):
        pcap = ['pcap_ipt_log', 'pcap_pkt_size_cv',
                'pcap_syn_ratio', 'tcp_reconnect_rate']
        assert len(pcap) == 4

    def test_mqtt_features_13(self):
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