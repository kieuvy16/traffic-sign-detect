from flask import Flask, request, render_template, jsonify, send_file
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import base64
import os
from werkzeug.utils import secure_filename
from gtts import gTTS
import pygame
import threading
import time
import tempfile

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load YOLO11 model
# MODEL_PATH = "src/models/yolo11/weights/best.pt"  # Adjust path to your trained model
# try:
#     if os.path.exists(MODEL_PATH):
#         model = YOLO(MODEL_PATH)
#         print(f"✅ Loaded custom model from: {MODEL_PATH}")
#     else:
#         # Fallback to pretrained model
#         print("⚠️ Custom model not found. Downloading pretrained YOLO11 model...")
#         print("✅ Loaded pretrained YOLO11n model")
    
#     # Test the model
#     print(f"📋 Model classes: {list(model.names.values())}")
    
# except Exception as e:
#     print(f"❌ Error loading model: {e}")
#     print("💡 Make sure you have internet connection for downloading pretrained model")
#     model = None
# MODEL_PATH = "src/models/yolo11/weights/best.pt"
MODEL_PATH = "runs/detect/train274/weights/best.pt"



try:
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print(f"✅ Loaded custom model from: {MODEL_PATH}")
    else:
        print("⚠️ Custom model not found. Downloading pretrained YOLO11 model...")
        # model = YOLO("yolo11n.pt")
        model = YOLO("runs/detect/train274/weights/best.pt")
        print("✅ Loaded pretrained YOLO11n model")

    print(f"📋 Model classes: {list(model.names.values())}")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("💡 Make sure you have internet connection for downloading pretrained model")
    model = None

# Initialize pygame mixer for audio playback
# os.environ["SDL_AUDIODRIVER"] = "directsound"
# pygame.mixer.init()
os.environ["SDL_AUDIODRIVER"] = "winmm"
# pygame.mixer.init()

# Vietnamese sign names mapping
SIGN_NAMES_VIETNAMESE = {
    '0verhead electrical cables': 'Cáp điện trên cao',
    'Bicycle ban': 'Cấm xe đạp',
    'Bus Stop': 'Trạm xe buýt',
    'Cars ban': 'Cấm ô tô',
    'Compulsary ahead': 'Bắt buộc đi thẳng',
    'Compulsory keep left': 'Bắt buộc rẽ trái',
    'Compulsory keep right': 'Bắt buộc rẽ phải',
    'Containers ban': 'Cấm xe container',
    'Dangerous Turn': 'Chỗ ngoặt nguy hiểm',
    'Left Turn': 'Rẽ trái',
    # 'Motobike ban': 'Cấm xe máy',
    'Motobike ban1': 'Cấm xe máy',
    'Motorcycles Only': 'Chỉ dành cho xe máy',
    'No Passenger Cars and Trucks': 'Cấm ô tô và xe tải',
    'No Two or Three-wheeled Vehicles': 'Cấm xe 2 hoặc 3 bánh',
    'No U-Turn and No turn right': 'Cấm quay đầu và rẽ phải',
    'No U-turn': 'Cấm quay đầu',
    'No U-turn No turn left': 'Cấm quay đầu và rẽ trái',
    'No car turn left': 'Cấm ô tô rẽ trái',
    'No car turn right': 'Cấm ô tô rẽ phải',
    'No parking': 'Cấm đỗ xe',
    'No parking stopping': 'Cấm dừng và đỗ xe',
    'No turn left': 'Cấm rẽ trái',
    'No turn right': 'Cấm rẽ phải',
    'One way': 'Đường một chiều',
    'Packing': 'Bãi đỗ xe',
    'Pedestrian crossing sign': 'Đường dành cho người đi bộ',
    'Pedestrians prohibited': 'Cấm người đi bộ',
    'Priority sign': 'Đường ưu tiên',
    'Prohibiting pedestrians': 'Cấm người đi bộ',
    'Slowly': 'Đi chậm',
    'Speed -limit 40': 'Giới hạn tốc độ 40',
    'Speed -limit 50': 'Giới hạn tốc độ 50',
    'Speed -limit 60': 'Giới hạn tốc độ 60',
    'Speed -limit 80': 'Giới hạn tốc độ 80',
    'U-Turn Allowed': 'Được phép quay đầu',
    'Watch for children': 'Chú ý trẻ em',
    'Yield sign': 'Nhường đường'
}

# Track announced signs with cooldown
announced_signs = {}  # signName -> lastAnnouncedTime
ANNOUNCE_COOLDOWN = 5  # 5 seconds cooldown
# is_speaking = False

# def speak_vietnamese(text_to_speak):
#     """Speak Vietnamese text using gTTS"""
#     global is_speaking
    
#     if is_speaking:
#         return
    
#     def _speak():
#         global is_speaking
#         is_speaking = True
#         try:
#             # Create TTS audio
#             tts = gTTS(text=text_to_speak, lang='vi')
            
#             # Save to temp file
#             with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
#                 temp_path = fp.name
#                 tts.save(temp_path)
            
#             # Play audio
#             pygame.mixer.music.load(temp_path)
#             pygame.mixer.music.play()
            
#             # Wait for audio to finish
#             while pygame.mixer.music.get_busy():
#                 time.sleep(0.1)
            
#             # Clean up temp file
#             try:
#                 os.unlink(temp_path)
#             except:
#                 pass
                
#         except Exception as e:
#             print(f"❌ TTS Error: {e}")
#         finally:
#             is_speaking = False
    
#     # Run in background thread
#     threading.Thread(target=_speak, daemon=True).start()
# def speak_vietnamese(text_to_speak):
#     global is_speaking

#     if is_speaking:
#         print("⏳ Đang nói, bỏ qua...")
#         return

#     def _speak():
#         global is_speaking
#         is_speaking = True
#         try:
#             print("🎤 Đang tạo TTS:", text_to_speak)

#             tts = gTTS(text=text_to_speak, lang='vi')

#             temp_path = os.path.join(tempfile.gettempdir(), "tts_audio.mp3")
#             tts.save(temp_path)

#             pygame.mixer.music.load(temp_path)
#             pygame.mixer.music.set_volume(1.0)
#             pygame.mixer.music.play()

#             while pygame.mixer.music.get_busy():
#                 time.sleep(0.1)

#             print("✔️ Đã đọc xong!")

#         except Exception as e:
#             print("❌ Lỗi TTS:", e)

#         finally:
#             is_speaking = False

#     threading.Thread(target=_speak, daemon=True).start()


# def announce_detection(class_name):
#     """Announce detection if not recently announced"""
#     global announced_signs
    
#     now = time.time()
#     last_announced = announced_signs.get(class_name, 0)
    
#     # Check cooldown
#     if now - last_announced < ANNOUNCE_COOLDOWN:
#         return
    
#     # Get Vietnamese name
#     vietnamese_name = SIGN_NAMES_VIETNAMESE.get(class_name, class_name)
#     text_to_speak = f"Phía trước có biển báo: {vietnamese_name}"
#     speak_vietnamese(text_to_speak)
    
#     # # Speak it
#     # speak_vietnamese(text_to_speak)
    
#     # Update last announced time
#     announced_signs[class_name] = now
# def speak_vietnamese(text):
#     try:
#         print("🎤 Tạo file TTS...")

#         # tạo file tạm nhưng KHÔNG giữ handle
#         temp_file = os.path.join(tempfile.gettempdir(), f"tts_{int(time.time()*1000)}.mp3")

#         tts = gTTS(text=text, lang="vi")
#         tts.save(temp_file)

#         print("🎧 Đang phát âm thanh…", temp_file)

#         pygame.mixer.music.load(temp_file)
#         pygame.mixer.music.play()

#         while pygame.mixer.music.get_busy():
#             time.sleep(0.1)

#         pygame.mixer.music.unload()   # 🔥 giải phóng file
#         os.remove(temp_file)          # 🔥 lúc này mới xóa an toàn

#         print("✔️ Phát xong & dọn file")

#     except Exception as e:
#         print("❌ TTS Error:", e)

#         is_speaking = False

#     threading.Thread(target=_speak, daemon=True).start()

is_speaking = False

# def speak_vietnamese(text):
#     global is_speaking

#     if is_speaking:
#         print("⏳ Đang nói, bỏ qua...")
#         return

#     def worker():
#         global is_speaking
     
#         try:   
#             is_speaking = True
#             print("🎤 Tạo file TTS...")

#             temp_file = os.path.join(
#                 tempfile.gettempdir(), 
#                 f"tts_{int(time.time()*1000)}.mp3"
#             )
            
#             pygame.mixer.init()

#             tts = gTTS(text=text, lang="vi")
#             tts.save(temp_file)

#             print("🎧 Đang phát âm thanh:", temp_file)

#             pygame.mixer.music.load(temp_file)
#             pygame.mixer.music.play()

#             while pygame.mixer.music.get_busy():
#                 time.sleep(0.1)
                
#             pygame.mixer.music.stop()
#             pygame.mixer.quit()

#             pygame.mixer.music.unload()
#             os.remove(temp_file)

#             print("✔️ Đã đọc")

#         except Exception as e:
#             print("❌ TTS Error:", e)

#         finally:
#             is_speaking = False

#     threading.Thread(target=worker, daemon=True).start()
def speak_vietnamese(text):
    global is_speaking

    if is_speaking:
        return

    def worker():
        global is_speaking
        is_speaking = True
        try:
            temp_file = os.path.join(
                tempfile.gettempdir(), 
                f"tts_{int(time.time()*1000)}.mp3"
            )

            tts = gTTS(text=text, lang="vi")
            tts.save(temp_file)

            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

            pygame.mixer.music.unload()
            os.remove(temp_file)

        except Exception as e:
            print("❌ TTS Error:", e)

        finally:
            is_speaking = False

    threading.Thread(target=worker, daemon=True).start()


def announce_detection(class_name):
    global announced_signs
    now = time.time()
    last = announced_signs.get(class_name, 0)

    if now - last < ANNOUNCE_COOLDOWN:
        print("⏸ Bỏ vì cooldown")
        return

    vn = SIGN_NAMES_VIETNAMESE.get(class_name, class_name)
    text = f"Phía trước có biển báo: {vn}"
    print("🔊 Sắp đọc:", text)

    speak_vietnamese(text)

    announced_signs[class_name] = now


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image):
    """Process image and return detection results"""
    try:
        print(f"📸 Processing image: {image.size}, mode: {image.mode}")
        
        # Ensure model is loaded
        if 'model' not in globals():
            raise Exception("Model not loaded")
        
        # Convert PIL image to OpenCV format
        img_array = np.array(image)
        print(f"🔄 Image array shape: {img_array.shape}")
        
        # Handle different image modes
        if image.mode == 'RGBA':
            # Convert RGBA to RGB
            image = image.convert('RGB')
            img_array = np.array(image)
        elif image.mode == 'L':
            # Convert grayscale to RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        print(f"🔄 Final array shape for inference: {img_array.shape}")
        
        # Run inference
        print("🤖 Running YOLO11 inference...")
        results = model(img_array)
        print(f"✅ Inference completed. Results: {len(results)}")
        
        # Check if results exist and have boxes
        if not results or len(results) == 0:
            raise Exception("No results returned from model")
        
        # Draw bounding boxes
        try:
            annotated_img = results[0].plot()
            print("✅ Bounding boxes drawn successfully")
        except Exception as plot_error:
            print(f"⚠️ Error drawing boxes: {plot_error}")
            # If plot fails, use original image
            annotated_img = img_array
        
        # Convert back to RGB for web display
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        annotated_pil = Image.fromarray(annotated_img)
        
        # Convert to base64 for web display
        buffer = io.BytesIO()
        annotated_pil.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        print("✅ Image converted to base64")
        
        # Extract detection info
        detections = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            print(f"📋 Found {len(results[0].boxes)} detections")
            for i, box in enumerate(results[0].boxes):
                try:
                    class_name =model.names[int(box.cls[0])]
                    vn_name = SIGN_NAMES_VIETNAMESE.get(class_name, class_name)
                    detection = {
                        'class_name': vn_name,
                        
                        'confidence': float(box.conf[0]),
                        'bbox': box.xyxy[0].tolist()  # [x1, y1
                        
                    }
                    announce_detection(class_name)

                    detections.append(detection)
                    print(f" Biển báo {i+1}: {vn_name} ({detection['confidence']:.2f})")
                except Exception as box_error:
                    print(f"⚠️ Error processing box {i}: {box_error}")
                    continue
        else:
            print("ℹ️ No detections found")
        
        return img_str, detections
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None, []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            print(f"📁 Processing file: {file.filename}")
            
            # Open image with better error handling
            try:
                image = Image.open(file.stream)
                print(f"📸 Image opened: {image.size}, {image.mode}")
            except Exception as img_error:
                print(f"❌ Error opening image: {img_error}")
                return jsonify({'error': f'Invalid image file: {str(img_error)}'}), 400
            
            # Process image
            result_img, detections = process_image(image)
            
            if result_img is None:
                return jsonify({'error': 'Failed to process image. Check server logs for details.'}), 500
            
            print(f"✅ Successfully processed image with {len(detections)} detections")
            
            return jsonify({
                'success': True,
                'image': result_img,
                'detections': detections,
                'count': len(detections)
            })
            
        except Exception as e:
            print(f"❌ Unexpected error in detect route: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type. Supported: PNG, JPG, JPEG, GIF, BMP'}), 400

@app.route('/speak', methods=['POST'])
def speak():
    """Endpoint to speak detected sign names"""
    try:
        data = request.get_json()
        class_name = data.get('class_name', '')
        confidence = data.get('confidence', 0)
        
        if class_name and confidence > 0.5:
            announce_detection(class_name)
            return jsonify({'success': True})
        
        return jsonify({'success': False, 'reason': 'Low confidence or no class name'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("🔊 TTS Engine: gTTS (Google Text-to-Speech)")
    app.run(debug=False, host='0.0.0.0', port=8000)
    