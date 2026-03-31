
# import cv2
# import os
# import time
# from ultralytics import YOLO

# # ================= CONFIG =================
# MODEL_PATH = "runs/detect/train274/weights/best.pt"

# # ================= LOAD MODEL =================
# def load_model():
#     try:
#         if os.path.exists(MODEL_PATH):
#             model = YOLO(MODEL_PATH)
#             print(f"✅ Load model: {MODEL_PATH}")
#         else:
#             print("⚠️ Không tìm thấy model → dùng pretrained")
#             model = YOLO("yolo11n.pt")

#         print(f"📋 Classes: {list(model.names.values())}")
#         return model

#     except Exception as e:
#         print(f"❌ Lỗi load model: {e}")
#         return None


# # ================= CAMERA =================
# def run_camera():
#     model = load_model()
#     if model is None:
#         return

#     cap = cv2.VideoCapture(0)

#     if not cap.isOpened():
#         print("❌ Không mở được camera")
#         return

#     print("📷 Camera đang chạy... (Q để thoát)")

#     prev_time = time.time()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # YOLO
#         results = model(frame, conf=0.15, imgsz=416)
#         annotated = results[0].plot()

#         # FPS thật
#         curr_time = time.time()
#         fps = 1 / (curr_time - prev_time)
#         prev_time = curr_time

#         cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         cv2.imshow("🚦 Camera Detection", annotated)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     print("✅ Đã tắt camera")


# # ================= VIDEO =================
# def run_video(video_path):
#     model = load_model()
#     if model is None:
#         return

#     if not os.path.exists(video_path):
#         print("❌ Không tìm thấy video")
#         return

#     cap = cv2.VideoCapture(video_path)

#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     out = cv2.VideoWriter(
#         "output.mp4",
#         cv2.VideoWriter_fourcc(*"mp4v"),
#         fps,
#         (640, 360)
#     )

#     print("🎥 Đang xử lý video...")

#     frame_count = 0
#     prev_time = time.time()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_count += 1

#         # 👉 Bỏ bớt frame để tăng tốc
#         if frame_count % 3 != 0:
#             continue

#         # 👉 Resize nhẹ
#         frame = cv2.resize(frame, (640, 360))

#         # YOLO
#         results = model(frame, conf=0.3, imgsz=416)
#         annotated = results[0].plot()

#         # FPS
#         curr_time = time.time()
#         fps_real = 1 / (curr_time - prev_time)
#         prev_time = curr_time

#         cv2.putText(annotated, f"FPS: {fps_real:.1f}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         out.write(annotated)

#         # 👉 mở nếu muốn xem realtime
#         cv2.imshow("🎥 Video Detection", annotated)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

#     print("✅ Xuất video: output.mp4")


# # ================= MENU =================
# if __name__ == "__main__":
#     print("===== 🚦 TRAFFIC SIGN DETECTION =====")
#     print("1. Camera")
#     print("2. Video")

#     choice = input("👉 Chọn mode: ")

#     if choice == "1":
#         run_camera()

#     elif choice == "2":
#         path = input("👉 Nhập đường dẫn video: ")
#         run_video(path)

#     else:
#         print("❌ Lựa chọn không hợp lệ")
import os
import time
import cv2
from ultralytics import YOLO

# ================= CONFIG =================
MODEL_PATH = "runs/detect/train274/weights/best.pt"
FALLBACK_MODEL = "yolo11n.pt"
OUTPUT_VIDEO_PATH = "output.mp4"

# Nếu muốn tăng tốc video thì đặt 1, 2, 3...
# 1 = xử lý mọi frame
FRAME_SKIP = 1

# Kích thước xử lý
PROCESS_WIDTH = 640
PROCESS_HEIGHT = 360


# ================= LOAD MODEL =================
def load_model():
    try:
        if os.path.exists(MODEL_PATH):
            loaded_model = YOLO(MODEL_PATH)
            print(f"✅ Load model: {MODEL_PATH}")
        else:
            print("⚠️ Không tìm thấy model custom → dùng pretrained")
            loaded_model = YOLO(FALLBACK_MODEL)
            print(f"✅ Load fallback model: {FALLBACK_MODEL}")

        print(f"📋 Classes: {list(loaded_model.names.values())}")
        return loaded_model

    except Exception as e:
        print(f"❌ Lỗi load model: {e}")
        return None


model = load_model()


# ================= HELPERS =================
def draw_fps(frame, fps_value):
    cv2.putText(
        frame,
        f"FPS: {fps_value:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


def infer_frame(frame, conf=0.25, imgsz=416):
    results = model(frame, conf=conf, imgsz=imgsz, verbose=False)
    annotated = results[0].plot()
    return annotated, results


# ================= CAMERA =================
def run_camera():
    if model is None:
        print("❌ Không thể chạy vì model chưa load được")
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Không mở được camera")
        return

    print("📷 Camera đang chạy... (Q để thoát)")

    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Không đọc được frame từ camera")
                break

            annotated, _ = infer_frame(frame, conf=0.15, imgsz=416)

            curr_time = time.time()
            delta = max(curr_time - prev_time, 1e-6)
            fps_real = 1.0 / delta
            prev_time = curr_time

            draw_fps(annotated, fps_real)

            cv2.imshow("Traffic Sign Detection - Camera", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Đã tắt camera")


# ================= VIDEO =================
def run_video(video_path):
    if model is None:
        print("❌ Không thể chạy vì model chưa load được")
        return

    if not os.path.exists(video_path):
        print(f"❌ Không tìm thấy video: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Không mở được video: {video_path}")
        return

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps <= 0:
        input_fps = 25.0

    # Nếu skip frame thì giảm fps output để giữ gần đúng thời lượng video
    output_fps = input_fps / FRAME_SKIP if FRAME_SKIP > 0 else input_fps
    output_fps = max(output_fps, 1.0)

    out = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (PROCESS_WIDTH, PROCESS_HEIGHT)
    )

    if not out.isOpened():
        cap.release()
        print("❌ Không tạo được file output.mp4")
        return

    print("🎥 Đang xử lý video...")

    frame_count = 0
    processed_count = 0
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if FRAME_SKIP > 1 and frame_count % FRAME_SKIP != 0:
                continue

            frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))

            annotated, _ = infer_frame(frame, conf=0.30, imgsz=416)

            curr_time = time.time()
            delta = max(curr_time - prev_time, 1e-6)
            fps_real = 1.0 / delta
            prev_time = curr_time

            draw_fps(annotated, fps_real)

            out.write(annotated)
            processed_count += 1

            cv2.imshow("Traffic Sign Detection - Video", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    print(f"✅ Đã xử lý {processed_count} frame")
    print(f"✅ Xuất video: {OUTPUT_VIDEO_PATH}")


# ================= MENU =================
if __name__ == "__main__":
    print("===== TRAFFIC SIGN DETECTION =====")
    print("1. Camera")
    print("2. Video")

    choice = input("👉 Chọn mode: ").strip()

    if choice == "1":
        run_camera()
    elif choice == "2":
        path = input("👉 Nhập đường dẫn video: ").strip()
        run_video(path)
    else:
        print("❌ Lựa chọn không hợp lệ")