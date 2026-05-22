import os
import json
import base64
import socket
import qrcode
import io
from datetime import date
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "aquarium_data.json")

def find_best_model():
    model_dir = os.path.join(BASE_DIR, "models")
    for name in ["betta_classifier-v5", "betta_classifier-v4", "betta_classifier-v3", "betta_classifier-final", "betta_classifier-2", "betta_classifier"]:
        pt = os.path.join(model_dir, name, "weights", "best.pt")
        if os.path.exists(pt):
            return pt
    return None

CLASS_THAI = {
    "healthy":    "ปลาสุขภาพดี",
    "fin_rot":    "ครีบเปื่อย (Fin Rot)",
    "fungus":     "เชื้อรา (Fungus)",
    "dropsy":     "ท้องมาน (Dropsy)",
    "white_spot": "จุดขาว (White Spot)",
}

TREATMENTS = {
    "healthy": {
        "color": "#A6E3A1",
        "icon": "✅",
        "cause": "ปลาแข็งแรง ไม่พบความผิดปกติ",
        "steps": [
            "รักษาอุณหภูมิน้ำ 26-30°C",
            "เปลี่ยนน้ำทุก 7 วัน 30-50%",
            "ให้อาหารวันละ 2 ครั้ง ปริมาณเหมาะสม",
            "หลีกเลี่ยงการเปิดไฟสว่างนานเกินไป"
        ]
    },
    "fin_rot": {
        "color": "#F38BA8",
        "icon": "🔴",
        "cause": "เกิดจากแบคทีเรีย Pseudomonas / Aeromonas — มักพบเมื่อน้ำสกปรกหรือปลาเครียด",
        "steps": [
            "แยกปลาออกจากตู้รวมทันที",
            "เปลี่ยนน้ำ 50% ทันที",
            "ใส่เกลือสินเธาว์ 1 ช้อนชา ต่อน้ำ 4 ลิตร",
            "ใช้ยา Methylene Blue หรือ Tetracycline ตามฉลาก",
            "รักษาอุณหภูมิ 28-30°C ตลอดการรักษา",
            "สังเกตอาการ 7-14 วัน"
        ]
    },
    "fungus": {
        "color": "#FAB387",
        "icon": "🟠",
        "cause": "เกิดจากเชื้อรา Saprolegnia — มักพบเมื่อปลามีบาดแผลหรืออุณหภูมิน้ำต่ำ",
        "steps": [
            "แยกปลาออกทันที",
            "ใช้ยา Malachite Green (0.1 ppm) หรือ Methylene Blue",
            "เปลี่ยนน้ำ 30% ทุกวันระหว่างรักษา",
            "รักษาอุณหภูมิ 27-28°C",
            "หลีกเลี่ยงการจับปลาบ่อยเพื่อลดบาดแผลใหม่",
            "รักษาต่อเนื่อง 7-10 วัน"
        ]
    },
    "dropsy": {
        "color": "#F38BA8",
        "icon": "🔴",
        "cause": "เกิดจากการติดเชื้อแบคทีเรียในอวัยวะภายใน ทำให้ท้องบวม เกล็ดตั้ง",
        "steps": [
            "แยกปลาออกทันที — โรคนี้รุนแรงและรักษายาก",
            "ใส่เกลือ Epsom salt 1/8 ช้อนชา ต่อ 5 ลิตร",
            "ใช้ยาปฏิชีวนะ Kanamycin หรือ Nitrofurazone",
            "เปลี่ยนน้ำ 25% ทุกวัน",
            "ปรึกษาสัตวแพทย์หากอาการไม่ดีขึ้นใน 3 วัน",
            "หากปลาไม่ทานอาหาร โอกาสรอดน้อยมาก"
        ]
    },
    "white_spot": {
        "color": "#F9E2AF",
        "icon": "🟡",
        "cause": "เกิดจากปรสิต Ichthyophthirius multifiliis (Ich) — แพร่กระจายเร็วมาก",
        "steps": [
            "เพิ่มอุณหภูมิน้ำเป็น 30°C (ปรสิตทนความร้อนไม่ได้)",
            "ใช้ยา Malachite Green + Formalin ตามฉลาก",
            "เปลี่ยนน้ำ 25% ทุก 2 วัน",
            "รักษา 10-14 วัน จนไม่เห็นจุดขาว",
            "ฆ่าเชื้อทุกอุปกรณ์ในตู้",
            "แยกปลาทุกตัวในตู้รวมออกด้วย"
        ]
    }
}

app = Flask(__name__)
model = None

def load_model():
    global model
    if not YOLO_AVAILABLE:
        return
    pt = find_best_model()
    if pt:
        model = YOLO(pt)
        print(f"โหลดโมเดล: {pt}")
    else:
        print("ไม่พบโมเดล — กรุณา train ก่อน")

load_model()

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "temperature": 28.0,
        "last_water_change": str(date.today()),
        "change_interval_days": 7,
        "fish_name": "ปลากัดของฉัน"
    }

def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"

@app.route("/")
def index():
    data = load_data()
    last_change = date.fromisoformat(data["last_water_change"])
    days_since = (date.today() - last_change).days
    days_left = max(0, data["change_interval_days"] - days_since)
    model_path = find_best_model() or "ยังไม่มีโมเดล"
    return render_template("index.html",
        data=data,
        days_since=days_since,
        days_left=days_left,
        treatments=TREATMENTS,
        class_thai=CLASS_THAI,
        model_loaded=(model is not None),
        model_path=model_path,
    )

@app.route("/api/status")
def api_status():
    data = load_data()
    last_change = date.fromisoformat(data["last_water_change"])
    days_since = (date.today() - last_change).days
    days_left = max(0, data["change_interval_days"] - days_since)
    return jsonify({**data, "days_since": days_since, "days_left": days_left,
                    "model_loaded": model is not None})

@app.route("/api/temperature", methods=["POST"])
def api_temperature():
    data = load_data()
    data["temperature"] = float(request.json["temperature"])
    save_data(data)
    return jsonify({"ok": True})

@app.route("/api/water-change", methods=["POST"])
def api_water_change():
    data = load_data()
    data["last_water_change"] = str(date.today())
    if "interval" in request.json:
        data["change_interval_days"] = int(request.json["interval"])
    save_data(data)
    return jsonify({"ok": True, "last_water_change": data["last_water_change"]})

@app.route("/api/detect", methods=["POST"])
def api_detect():
    if model is None:
        return jsonify({"error": "ยังไม่มีโมเดล กรุณารัน train_model.py ก่อน"}), 503

    try:
        img_b64 = request.json["image"]
        if "," in img_b64:
            img_b64 = img_b64.split(",")[1]
        img_bytes = base64.b64decode(img_b64)
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "อ่านภาพไม่ได้"}), 400

        # ลดขนาดให้ไม่เกิน 320px เพื่อให้ประมวลผลเร็วขึ้น
        h, w = img.shape[:2]
        if max(h, w) > 320:
            scale = 320 / max(h, w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)))

        tmp = os.path.join(BASE_DIR, "_tmp_web.jpg")
        cv2.imwrite(tmp, img)

        results = model(tmp, verbose=False, imgsz=224)
        probs = results[0].probs
        names = results[0].names

        top1_idx  = probs.top1
        top1_conf = float(probs.top1conf)
        class_key = names[top1_idx]

        top5 = [
            {"class": names[i], "thai": CLASS_THAI.get(names[i], names[i]), "conf": round(float(c), 4)}
            for i, c in zip(probs.top5, probs.top5conf.tolist())
        ]

        return jsonify({
            "class": class_key,
            "thai": CLASS_THAI.get(class_key, class_key),
            "confidence": round(top1_conf, 4),
            "top5": top5,
            "treatment": TREATMENTS.get(class_key, {}),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/qr")
def api_qr():
    render_url = os.environ.get("RENDER_EXTERNAL_URL")
    if render_url:
        url = render_url
    else:
        ip = get_local_ip()
        port = int(os.environ.get("PORT", 5000))
        url = f"http://{ip}:{port}"
    qr = qrcode.QRCode()
    qr.add_data(url)
    qr.make(fit=True)
    qr_img = qr.make_image()
    buf = io.BytesIO()
    qr_img.save(buf)
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return jsonify({"url": url, "qr": f"data:image/png;base64,{b64}"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    ip = get_local_ip()
    print(f"\n{'='*50}")
    print(f"เปิด Web App ที่: http://{ip}:{port}")
    print(f"สแกน QR code บนมือถือ (WiFi เดียวกัน)")
    print(f"{'='*50}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
