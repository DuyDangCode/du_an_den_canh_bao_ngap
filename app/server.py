import http.server
import json
import os
import sys
import urllib.parse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.ml_engine import FloodMLEngine

PORT = int(os.environ.get("PORT", 8000))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
STATIC_DIR = os.path.join(BASE_DIR, "static")
STATE_FILE = os.path.join(BASE_DIR, "data", "model_state.json")

ml_engine = FloodMLEngine(state_file=STATE_FILE)

# Cấu hình ngưỡng mặc định
default_thresholds = {
    "c1_rainfall": 40.0,
    "c1_tide": 140.0,
    "c2_drain": 70.0,
    "c2_drain_speed": 2.0,
    "c3_floor_water": 1.0,
    "anti_false_alarm_seconds": 30.0,
    "house_elevation": 20.0,
    "lamp_height": 80.0
}

current_thresholds = dict(default_thresholds)

class AppRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/status":
            self.send_json_response({
                "weights": ml_engine.weights,
                "episodes": ml_engine.training_episodes,
                "thresholds": current_thresholds,
                "history": ml_engine.history[-10:]
            })
            return

        # Phục vụ file STL
        if path == "/resources/house.stl":
            stl_path = os.path.join(PROJECT_ROOT, "resources", "house.stl")
            if os.path.exists(stl_path):
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(os.path.getsize(stl_path)))
                self.end_headers()
                with open(stl_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_error(404, "house.stl not found")
                return

        # Phục vụ file tĩnh
        if path == "/" or path == "/index.html":
            file_path = os.path.join(STATIC_DIR, "index.html")
        else:
            rel_path = path.lstrip("/")
            file_path = os.path.join(STATIC_DIR, rel_path)

        if os.path.exists(file_path) and os.path.isfile(file_path):
            content_type = "text/html"
            if file_path.endswith(".js"):
                content_type = "application/javascript"
            elif file_path.endswith(".css"):
                content_type = "text/css"
            elif file_path.endswith(".json"):
                content_type = "application/json"
            elif file_path.endswith(".stl"):
                content_type = "application/octet-stream"

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(os.path.getsize(file_path)))
            self.end_headers()
            with open(file_path, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404, "File not found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode("utf-8")) if post_data else {}

        if path == "/api/predict":
            features = data.get("features", {})
            # Cập nhật thông số cốt nền & chiều cao đèn từ thresholds nếu features chưa truyền
            if "house_elevation" not in features:
                features["house_elevation"] = current_thresholds.get("house_elevation", 20.0)
            if "lamp_height" not in features:
                features["lamp_height"] = current_thresholds.get("lamp_height", 80.0)

            result = ml_engine.predict(features, current_thresholds)
            self.send_json_response(result)

        elif path == "/api/train_online":
            features = data.get("features", {})
            actual_cm = float(data.get("actual_cm", 0.0))
            if "house_elevation" not in features:
                features["house_elevation"] = current_thresholds.get("house_elevation", 20.0)
            if "lamp_height" not in features:
                features["lamp_height"] = current_thresholds.get("lamp_height", 80.0)

            train_result = ml_engine.train_online(features, actual_cm)
            self.send_json_response({
                "success": True,
                "record": train_result,
                "weights": ml_engine.weights,
                "episodes": ml_engine.training_episodes
            })

        elif path == "/api/update_thresholds":
            new_th = data.get("thresholds", {})
            for k in current_thresholds:
                if k in new_th:
                    current_thresholds[k] = float(new_th[k])
            self.send_json_response({
                "success": True,
                "thresholds": current_thresholds
            })
        else:
            self.send_error(404, "API endpoint not found")

    def send_json_response(self, data):
        response_bytes = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

def run_server():
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    server_address = ("", PORT)
    httpd = http.server.HTTPServer(server_address, AppRequestHandler)
    print(f"🚀 Server đang chạy tại http://localhost:{PORT}")
    httpd.serve_forever()

if __name__ == "__main__":
    run_server()
