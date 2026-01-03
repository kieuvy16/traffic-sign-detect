"""
Traffic Sign Detection using Laptop Camera
Sử dụng YOLO11 để nhận diện biển báo giao thông qua webcam laptop
"""

import cv2
import os
from ultralytics import YOLO

# Load YOLO11 model
MODEL_PATH = "src/models/yolo11/weights/best.pt"

def load_model():
    """Load trained YOLO11 model"""
    try:
        # Thử load từ đường dẫn tương đối trước
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
            print(f"✅ Đã load model từ: {MODEL_PATH}")
        elif os.path.exists("models/yolo11/weights/best.pt"):
            model = YOLO("models/yolo11/weights/best.pt")
            print("✅ Đã load model từ: models/yolo11/weights/best.pt")
        else:
            print("⚠️ Không tìm thấy model custom. Đang load model pretrained...")
            model = YOLO("yolo11n.pt")
            print("✅ Đã load model YOLO11n pretrained")
        
        print(f"📋 Classes của model: {list(model.names.values())}")
        return model
        
    except Exception as e:
        print(f"❌ Lỗi khi load model: {e}")
        return None

def run_camera_detection():
    """Chạy detection từ webcam laptop"""
    
    # Load model
    model = load_model()
    if model is None:
        print("❌ Không thể khởi động vì model chưa được load!")
        return
    
    # Mở camera
    print("📷 Đang mở camera...")
    cap = cv2.VideoCapture(0)  # 0 = camera mặc định của laptop
    
    if not cap.isOpened():
        print("❌ Không thể mở camera! Vui lòng kiểm tra:")
        print("   - Camera có đang được sử dụng bởi ứng dụng khác không?")
        print("   - Driver camera đã được cài đặt chưa?")
        return
    
    # Cấu hình camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✅ Camera đã sẵn sàng!")
    print("=" * 50)
    print("📌 HƯỚNG DẪN SỬ DỤNG:")
    print("   - Nhấn 'Q' để thoát")
    print("   - Nhấn 'S' để chụp và lưu ảnh hiện tại")
    print("   - Nhấn 'P' để tạm dừng/tiếp tục")
    print("=" * 50)
    
    paused = False
    frame_count = 0
    screenshot_count = 0
    
    # Tạo thư mục lưu ảnh nếu chưa có
    os.makedirs("captured_images", exist_ok=True)
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ Không thể đọc frame từ camera!")
                    break
                
                frame_count += 1
                
                # Chạy detection
                results = model(frame, conf=0.5, verbose=False)
                
                # Vẽ kết quả lên frame
                annotated_frame = results[0].plot()
                
                # Hiển thị thông tin detection
                detections = results[0].boxes
                if detections is not None and len(detections) > 0:
                    for box in detections:
                        class_name = model.names[int(box.cls)]
                        confidence = float(box.conf)
                        print(f"🚦 Phát hiện: {class_name} - Độ tin cậy: {confidence:.2%}")
                
                # Thêm thông tin lên màn hình
                cv2.putText(annotated_frame, f"FPS: {cap.get(cv2.CAP_PROP_FPS):.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Phat hien: {len(detections) if detections is not None else 0}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated_frame, "Nhan Q de thoat | S: Chup | P: Tam dung", 
                           (10, annotated_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            else:
                # Khi tạm dừng, hiển thị thông báo
                cv2.putText(annotated_frame, "TAM DUNG - Nhan P de tiep tuc", 
                           (annotated_frame.shape[1]//2 - 200, annotated_frame.shape[0]//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Hiển thị frame
            cv2.imshow("Traffic Sign Detection - Camera", annotated_frame)
            
            # Xử lý phím bấm
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("👋 Đang thoát...")
                break
            elif key == ord('s') or key == ord('S'):
                # Lưu ảnh
                screenshot_count += 1
                filename = f"captured_images/screenshot_{screenshot_count}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"📸 Đã lưu ảnh: {filename}")
            elif key == ord('p') or key == ord('P'):
                paused = not paused
                if paused:
                    print("⏸️ Đã tạm dừng")
                else:
                    print("▶️ Tiếp tục...")
                    
    except KeyboardInterrupt:
        print("\n👋 Đã ngắt bởi người dùng!")
        
    finally:
        # Giải phóng camera và đóng cửa sổ
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Đã đóng camera và giải phóng tài nguyên")
        print(f"📊 Tổng số frame đã xử lý: {frame_count}")

if __name__ == "__main__":
    print("=" * 50)
    print("🚦 TRAFFIC SIGN DETECTION - CAMERA MODE")
    print("=" * 50)
    run_camera_detection()
