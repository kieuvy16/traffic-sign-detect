
from flask import Flask, request, render_template, jsonify
import base64
import io
import os
import tempfile
import threading
import time
from collections import Counter

import cv2
import numpy as np
import pygame
from gtts import gTTS
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

MODEL_PATH = "runs/detect/train274/weights/best.pt"
FALLBACK_MODEL = "yolo11n.pt"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp"}

detection_history = []
announced_signs = {}
ANNOUNCE_COOLDOWN = 5  # seconds

tts_lock = threading.Lock()
mixer_ready = False


SIGN_NAMES_VIETNAMESE = {
    # I. BIỂN CẤM (0–29)
    "No entry": "Cấm đi ngược chiều",
    "No cars": "Cấm ô tô",
    "No motorcycles": "Cấm xe mô tô",
    "No bicycles": "Cấm xe đạp",
    "No pedestrians": "Cấm người đi bộ",
    "No trucks": "Cấm xe tải",
    "No container vehicles": "Cấm xe container",
    "No U-turn": "Cấm quay đầu xe",
    "No stopping": "Cấm dừng xe",
    "No parking": "Cấm đỗ xe",
    "No stopping and parking": "Cấm dừng và đỗ xe",
    "No overtaking": "Cấm vượt",
    "No horn": "Cấm bấm còi",
    "No buses": "Cấm xe buýt",
    "No three-wheeled vehicles": "Cấm xe ba bánh",
    "No animal-drawn vehicles": "Cấm xe súc vật kéo",
    "Height limit": "Hạn chế chiều cao",
    "Width limit": "Hạn chế chiều rộng",
    "Length limit": "Hạn chế chiều dài",
    "Axle load limit": "Hạn chế tải trọng trục xe",
    "Vehicle weight limit": "Hạn chế trọng lượng xe",
    "No non-motorized vehicles": "Cấm xe thô sơ",
    "No tractors": "Cấm máy kéo",
    "No trailers": "Cấm rơ moóc",
    "No mopeds": "Cấm xe gắn máy",
    "No turn right": "Cấm rẽ phải",
    "No turn left": "Cấm rẽ trái",
    "No turn right and No U-turn": "Cấm rẽ phải và quay đầu xe",
    "No turn left and No U-turn": "Cấm rẽ trái và quay đầu xe",

    # II. BIỂN NGUY HIỂM (30–44)
    "Dangerous curve left": "Đường cong nguy hiểm bên trái",
    "Dangerous curve right": "Đường cong nguy hiểm bên phải",
    "Intersection ahead": "Giao nhau phía trước",
    "Traffic signals ahead": "Đèn tín hiệu phía trước",
    "Narrow road": "Đường hẹp",
    "Slippery road": "Đường trơn",
    "Steep ascent": "Dốc lên nguy hiểm",
    "Steep descent": "Dốc xuống nguy hiểm",
    "Two-way traffic": "Đường hai chiều",
    "Pedestrian crossing": "Đường người đi bộ",
    "Children crossing": "Trẻ em qua đường",
    "Road works": "Công trường thi công",
    "Animals crossing": "Động vật qua đường",
    "Railway crossing": "Giao nhau với đường sắt",
    "Falling rocks": "Đá rơi",

    # III. BIỂN HIỆU LỆNH & CHỈ DẪN (45–59)
    "Go straight": "Đi thẳng",
    "Turn left": "Rẽ trái",
    "Turn right": "Rẽ phải",
    "Go straight or left": "Đi thẳng hoặc rẽ trái",
    "Go straight or right": "Đi thẳng hoặc rẽ phải",
    "Roundabout": "Vòng xuyến",
    "Pedestrian path": "Đường dành cho người đi bộ",
    "Bicycle path": "Đường dành cho xe đạp",
    "Parking": "Bãi đỗ xe",
    "Bus stop": "Điểm dừng xe buýt",
    "Hospital": "Bệnh viện",
    "Gas station": "Trạm xăng",
    "Restaurant": "Nhà hàng",
    "Hotel": "Khách sạn",
    "Slow": "Đi chậm",
}


def load_model():
    try:
        if os.path.exists(MODEL_PATH):
            loaded_model = YOLO(MODEL_PATH)
            print(f"✅ Loaded custom model from: {MODEL_PATH}")
        else:
            print("⚠️ Custom model not found. Loading fallback model...")
            loaded_model = YOLO(FALLBACK_MODEL)
            print(f"✅ Loaded fallback model: {FALLBACK_MODEL}")

        print(f"📋 Model classes: {list(loaded_model.names.values())}")
        return loaded_model

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


model = load_model()


def init_audio():
    global mixer_ready

    if mixer_ready:
        return True

    try:
        if os.name == "nt":
            os.environ.setdefault("SDL_AUDIODRIVER", "winmm")

        pygame.mixer.init()
        mixer_ready = True
        print("🔊 Audio mixer initialized")
        return True

    except Exception as e:
        mixer_ready = False
        print(f"⚠️ Audio init failed: {e}")
        return False


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def speak_vietnamese(text):
    if not text:
        return

    if not init_audio():
        print("⚠️ Skip speaking because audio mixer is unavailable")
        return

    # tránh nhiều luồng cùng phát âm thanh một lúc
    if not tts_lock.acquire(blocking=False):
        print("⏳ TTS is busy, skipping new speech")
        return

    def worker():
        temp_file = None
        try:
            temp_file = os.path.join(
                tempfile.gettempdir(),
                f"tts_{int(time.time() * 1000)}.mp3"
            )

            tts = gTTS(text=text, lang="vi")
            tts.save(temp_file)

            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

        except Exception as e:
            print(f"❌ TTS Error: {e}")

        finally:
            try:
                if pygame.mixer.get_init():
                    try:
                        pygame.mixer.music.stop()
                    except Exception:
                        pass
                    try:
                        pygame.mixer.music.unload()
                    except Exception:
                        pass
            except Exception:
                pass

            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as cleanup_error:
                    print(f"⚠️ Could not remove temp audio file: {cleanup_error}")

            tts_lock.release()

    threading.Thread(target=worker, daemon=True).start()


# def announce_detection(class_name):
#     now = time.time()
#     last_announced = announced_signs.get(class_name, 0)

#     if now - last_announced < ANNOUNCE_COOLDOWN:
#         print("⏸ Bỏ vì cooldown")
#         return

#     vn_name = SIGN_NAMES_VIETNAMESE.get(class_name, class_name)
#     text = f"Phía trước có biển báo: {vn_name}"
#     print("🔊 Sắp đọc:", text)

#     speak_vietnamese(text)
#     announced_signs[class_name] = now
def announce_detection(sign_text):
    now = time.time()
    last_announced = announced_signs.get(sign_text, 0)

    if now - last_announced < ANNOUNCE_COOLDOWN:
        print("⏸ Bỏ vì cooldown")
        return

    text = f"Phía trước có biển báo: {sign_text}"
    print("🔊 Sắp đọc:", text)

    speak_vietnamese(text)
    announced_signs[sign_text] = now

def process_image(image):
    try:
        print(f"📸 Processing image: size={image.size}, mode={image.mode}")

        if model is None:
            raise Exception("Model not loaded")

        if image.mode != "RGB":
            image = image.convert("RGB")

        img_rgb = np.array(image)
        print(f"🔄 RGB image shape: {img_rgb.shape}")

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        print(f"🔄 BGR image shape for inference: {img_bgr.shape}")

        print("🤖 Running YOLO inference...")
        results = model(img_bgr)
        print(f"✅ Inference completed. Results count: {len(results)}")

        if not results or len(results) == 0:
            raise Exception("No results returned from model")

        result = results[0]

        try:
            annotated_bgr = result.plot()
            print("✅ Bounding boxes drawn successfully")
        except Exception as plot_error:
            print(f"⚠️ Error drawing boxes: {plot_error}")
            annotated_bgr = img_bgr.copy()

        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        annotated_pil = Image.fromarray(annotated_rgb)

        buffer = io.BytesIO()
        annotated_pil.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        detections = []

        if result.boxes is not None and len(result.boxes) > 0:
            print(f"📋 Found {len(result.boxes)} detections")

            for i, box in enumerate(result.boxes):
                try:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    vn_name = SIGN_NAMES_VIETNAMESE.get(class_name, class_name)
                    confidence = float(box.conf[0])

                    detection = {
                        "class_name": vn_name,
                        "confidence": confidence,
                        "bbox": box.xyxy[0].tolist(),
                    }

                    detections.append(detection)

                    if confidence > 0.5:
                        announce_detection(vn_name)
                        detection_history.append(vn_name)

                    print(f"✅ Biển báo {i + 1}: {vn_name} ({confidence:.2f})")

                except Exception as box_error:
                    print(f"⚠️ Error processing box {i}: {box_error}")

        else:
            print("ℹ️ No detections found")

        return img_str, detections

    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return None, []


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": "Invalid file type. Supported: PNG, JPG, JPEG, GIF, BMP"
        }), 400

    try:
        print(f"📁 Processing file: {file.filename}")

        try:
            image = Image.open(file.stream)
            image.load()  # force đọc ảnh để bắt lỗi sớm
            print(f"📸 Image opened: {image.size}, {image.mode}")
        except Exception as img_error:
            print(f"❌ Error opening image: {img_error}")
            return jsonify({"error": f"Invalid image file: {str(img_error)}"}), 400

        result_img, detections = process_image(image)

        if result_img is None:
            return jsonify({
                "error": "Failed to process image. Check server logs for details."
            }), 500

        print(f"✅ Successfully processed image with {len(detections)} detections")

        return jsonify({
            "success": True,
            "image": result_img,
            "detections": detections,
            "count": len(detections),
        })

    except Exception as e:
        print(f"❌ Unexpected error in detect route: {e}")
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500


@app.route("/history", methods=["GET"])
def get_history():
    count = Counter(detection_history)

    result = []
    for name, num in count.most_common():
        result.append({
            "name": name,
            "count": num
        })

    return jsonify({
        "total": len(detection_history),
        "data": result
    })


# @app.route("/speak", methods=["POST"])
# def speak():
#     """Speak sign name manually from client"""
#     try:
#         data = request.get_json(silent=True) or {}

#         class_name = str(data.get("class_name", "")).strip()
#         confidence = float(data.get("confidence", 0) or 0)

#         if not class_name:
#             return jsonify({"success": False, "reason": "No class name provided"}), 400

#         if confidence <= 0.5:
#             return jsonify({
#                 "success": False,
#                 "reason": "Low confidence"
#             })

#         announce_detection(class_name)

#         # Không append history ở đây để tránh bị nhân đôi
#         # vì /detect đã append rồi
#         return jsonify({"success": True})

#     except Exception as e:
#         return jsonify({"success": False, "error": str(e)}), 500
@app.route("/speak", methods=["POST"])
def speak():
    try:
        data = request.get_json(silent=True) or {}

        class_name = str(data.get("class_name", "")).strip()
        confidence = float(data.get("confidence", 0) or 0)

        if not class_name:
            return jsonify({"success": False, "reason": "No class name provided"}), 400

        if confidence <= 0.5:
            return jsonify({"success": False, "reason": "Low confidence"})

        announce_detection(class_name)
        return jsonify({"success": True})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)

    print("🔊 TTS Engine: gTTS (Google Text-to-Speech)")
    init_audio()
    app.run(debug=False, host="0.0.0.0", port=8000)