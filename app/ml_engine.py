"""
Bộ não Machine Learning Nhẹ Học Thích Nghi / Học Trực Tuyến (Online Learning Engine).
Sử dụng hoàn toàn Python Standard Library (không cần cài thêm numpy/sklearn).
Hỗ trợ:
- Huấn luyện nền (Base Training) từ dữ liệu mô phỏng ngập lịch sử TP.HCM.
- Dự báo tức thì độ sâu ngập dự kiến (cm) và xác suất ngập (%).
- Học trực tuyến (Online Gradient Descent / partial_fit) khi nhận phản hồi thực tế từ ngôi nhà.
- Bộ lọc chống báo động giả khi lau nhà hoặc làm đổ nước (ngưỡng 30s).
"""

import math
import json
import os

DEFAULT_WEIGHTS = {
    "bias": -5.0,
    "rainfall": 0.25,        # mm/h -> cm ngập
    "tide": 0.18,            # cm triều cường -> cm ngập
    "drain_level": 0.12,     # % cống đầy -> cm ngập
    "drain_speed": 1.20,     # cm/phút tốc độ nước dâng -> cm ngập
    "humidity": 0.05,        # % độ ẩm
    "pressure": -0.15,       # hPa áp suất giảm -> tăng nguy cơ
    "house_elevation": -0.85 # cm cốt nền nhà cao -> giảm ngập
}

class FloodMLEngine:
    def __init__(self, state_file=None):
        self.state_file = state_file
        self.weights = dict(DEFAULT_WEIGHTS)
        self.learning_rate = 0.01
        self.training_episodes = 0
        self.history = []
        self.load_state()

    def load_state(self):
        if self.state_file and os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.weights = data.get("weights", self.weights)
                    self.training_episodes = data.get("training_episodes", 0)
                    self.history = data.get("history", [])
            except Exception as e:
                print(f"[ML Engine] Lỗi khi đọc state_file: {e}")

    def save_state(self):
        if self.state_file:
            try:
                os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
                with open(self.state_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "weights": self.weights,
                        "training_episodes": self.training_episodes,
                        "history": self.history[-50:]
                    }, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[ML Engine] Lỗi khi lưu state_file: {e}")

    def predict(self, features, thresholds=None):
        """
        features: dict chứa:
          - rainfall (mm/h)
          - tide (cm)
          - humidity (%)
          - pressure (hPa)
          - drain_level (%)
          - drain_speed (cm/min)
          - floor_water (cm)
          - floor_water_duration (giây)
          - house_elevation (cm)
          - lamp_height (cm)
        """
        thresholds = thresholds or {}
        th_c1_rain = thresholds.get("c1_rainfall", 40.0)
        th_c1_tide = thresholds.get("c1_tide", 140.0)
        th_c2_drain = thresholds.get("c2_drain", 70.0)
        th_c2_speed = thresholds.get("c2_drain_speed", 2.0)
        th_c3_floor = thresholds.get("c3_floor_water", 1.0)
        th_filter_time = thresholds.get("anti_false_alarm_seconds", 30.0)

        r = float(features.get("rainfall", 0.0))
        t = float(features.get("tide", 0.0))
        h = float(features.get("humidity", 70.0))
        p = float(features.get("pressure", 1010.0)) - 1000.0  # normalize quanh 1000hPa
        dl = float(features.get("drain_level", 0.0))
        ds = float(features.get("drain_speed", 0.0))
        elev = float(features.get("house_elevation", 20.0))
        fw = float(features.get("floor_water", 0.0))
        f_dur = float(features.get("floor_water_duration", 0.0))
        lamp_height = float(features.get("lamp_height", 80.0))

        # Tính tuyến tính độ sâu ngập dự kiến
        raw_pred = (
            self.weights["bias"] +
            self.weights["rainfall"] * r +
            self.weights["tide"] * (t - 100.0) +
            self.weights["humidity"] * (h - 60.0) +
            self.weights["pressure"] * p +
            self.weights["drain_level"] * dl +
            self.weights["drain_speed"] * ds +
            self.weights["house_elevation"] * elev
        )

        predicted_depth = max(0.0, round(raw_pred, 1))

        # Tính xác suất ngập (%) qua hàm sigmoid
        z = (raw_pred - 2.0) / 5.0
        prob = 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, z))))
        probability_pct = round(prob * 100, 1)

        # Đóng góp của từng đặc trưng
        contributions = {
            "rainfall_contrib": round(self.weights["rainfall"] * r, 1),
            "tide_contrib": round(self.weights["tide"] * max(0.0, t - 100.0), 1),
            "drain_contrib": round(self.weights["drain_level"] * dl + self.weights["drain_speed"] * ds, 1),
            "elevation_offset": round(self.weights["house_elevation"] * elev, 1),
        }

        # Kiểm tra logic chống báo động giả khi lau nhà / làm đổ nước
        is_water_on_floor = fw >= th_c3_floor
        is_drain_full_or_raining = (dl >= th_c2_drain) or (r >= th_c1_rain)
        is_duration_passed = f_dur >= th_filter_time

        is_false_alarm = False
        false_alarm_reason = ""

        # Xác định Cấp độ cảnh báo (0, 1, 2, 3)
        if is_water_on_floor:
            if is_duration_passed and is_drain_full_or_raining:
                level = 3
                level_name = "Khẩn cấp (Nước tràn sàn nhà)"
            else:
                is_false_alarm = True
                if not is_duration_passed:
                    false_alarm_reason = f"Nước đọng mới {f_dur:.0f}s / ngưỡng {th_filter_time:.0f}s (đang lau nhà/đổ ly nước)"
                else:
                    false_alarm_reason = "Có nước trên sàn nhưng cống khô ráo & trời không mưa (nước sinh hoạt)"
                # Khi phát hiện báo giả, hạ cấp độ về theo dự báo cống/thời tiết
                if dl >= th_c2_drain or ds >= th_c2_speed or predicted_depth > 5.0:
                    level = 2
                    level_name = "Chuẩn bị ứng phó"
                elif r >= th_c1_rain or t >= th_c1_tide or probability_pct >= 40.0:
                    level = 1
                    level_name = "Dự báo sớm"
                else:
                    level = 0
                    level_name = "Ngày bình thường"
        elif dl >= th_c2_drain or ds >= th_c2_speed or predicted_depth > 5.0:
            level = 2
            level_name = "Chuẩn bị ứng phó"
        elif r >= th_c1_rain or t >= th_c1_tide or probability_pct >= 40.0:
            level = 1
            level_name = "Dự báo sớm"
        else:
            level = 0
            level_name = "Ngày bình thường"

        # Tính chiều cao cột sáng đèn (cm và % so với thân đèn)
        display_depth = fw if (level == 3 and not is_false_alarm) else predicted_depth
        display_depth = max(0.0, display_depth)
        ruler_pct = min(100.0, round((display_depth / max(10.0, lamp_height)) * 100, 1))

        return {
            "level": level,
            "level_name": level_name,
            "predicted_depth_cm": predicted_depth,
            "display_depth_cm": display_depth,
            "ruler_pct": ruler_pct,
            "probability_pct": probability_pct,
            "contributions": contributions,
            "is_false_alarm": is_false_alarm,
            "false_alarm_reason": false_alarm_reason,
            "zones": {
                "manhole": "Tràn đầy / Nguy cơ cao" if dl >= th_c2_drain else ("Đang dâng" if dl >= 40 else "Bình thường"),
                "living_room": "Bị ngập" if (level == 3 and not is_false_alarm) else ("Mấp mé / Nguy cơ" if level >= 2 else "Khô ráo"),
                "kitchen_toilet": "Bị ngập" if (level == 3 and not is_false_alarm) else ("Cảnh báo" if level >= 2 else "Khô ráo")
            }
        }

    def train_online(self, features, actual_flood_depth_cm):
        """
        Cập nhật học trực tuyến / tăng cường nhẹ (Online SGD partial_fit)
        features: đầu vào lúc cơn mưa diễn ra
        actual_flood_depth_cm: mức ngập thực tế mà chủ nhà đo được sau cơn mưa
        """
        pred_before = self.predict(features)
        y_pred = pred_before["predicted_depth_cm"]
        y_true = max(0.0, float(actual_flood_depth_cm))

        error = y_true - y_pred  # > 0: dự báo thiếu, < 0: dự báo thừa
        reward = max(-10.0, 10.0 - abs(error))

        # Vector đầu vào chuẩn hóa cho cập nhật Gradient Descent
        r = float(features.get("rainfall", 0.0))
        t = float(features.get("tide", 0.0)) - 100.0
        dl = float(features.get("drain_level", 0.0))
        ds = float(features.get("drain_speed", 0.0))
        elev = float(features.get("house_elevation", 20.0))

        lr = self.learning_rate
        # Gradient update với clipping để tránh bùng nổ
        grad_bias = max(-5.0, min(5.0, error))
        self.weights["bias"] += lr * grad_bias
        self.weights["rainfall"] += lr * 0.01 * error * max(0.0, r)
        self.weights["tide"] += lr * 0.01 * error * max(0.0, t)
        self.weights["drain_level"] += lr * 0.01 * error * dl
        self.weights["drain_speed"] += lr * 0.05 * error * ds
        self.weights["house_elevation"] += lr * 0.02 * (-error) * elev  # Cốt nền càng cao thì càng giảm

        # Giữ các trọng số trong giới hạn vật lý hợp lý
        self.weights["rainfall"] = max(0.01, min(1.0, self.weights["rainfall"]))
        self.weights["tide"] = max(0.01, min(1.0, self.weights["tide"]))
        self.weights["drain_level"] = max(0.01, min(0.5, self.weights["drain_level"]))
        self.weights["drain_speed"] = max(0.1, min(3.0, self.weights["drain_speed"]))
        self.weights["house_elevation"] = max(-2.0, min(-0.1, self.weights["house_elevation"]))

        self.training_episodes += 1
        pred_after = self.predict(features)

        record = {
            "episode": self.training_episodes,
            "actual_cm": y_true,
            "pred_before_cm": y_pred,
            "pred_after_cm": pred_after["predicted_depth_cm"],
            "error_before": round(abs(error), 2),
            "error_after": round(abs(y_true - pred_after["predicted_depth_cm"]), 2),
            "reward": round(reward, 2)
        }
        self.history.append(record)
        self.save_state()

        return record
