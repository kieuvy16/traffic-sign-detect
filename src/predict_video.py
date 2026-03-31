# import cv2
# from ultralytics import YOLO

# # Load model
# MODEL_PATH = "runs/detect/train274/weights/best.pt"

# # Đường dẫn video input
# video_path = "input.mp4"

# # Đọc video
# cap = cv2.VideoCapture(video_path)

# # Lấy thông số video
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # Tạo video output
# out = cv2.VideoWriter(
#     "output.mp4",
#     cv2.VideoWriter_fourcc(*"mp4v"),
#     fps,
#     (width, height)
# )

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Detect
#     results = model(frame)

#     # Vẽ kết quả
#     annotated_frame = results[0].plot()

#     # Ghi video
#     out.write(annotated_frame)

#     # Hiển thị (optional)
#     cv2.imshow("Video Detection", annotated_frame)
#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cap.release()
# out.release()
# cv2.destroyAllWindows()
import os
import cv2
from ultralytics import YOLO

# Load model
MODEL_PATH = "runs/detect/train274/weights/best.pt"
VIDEO_PATH = "input.mp4"
OUTPUT_PATH = "output.mp4"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Không tìm thấy model: {MODEL_PATH}")

if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"Không tìm thấy video input: {VIDEO_PATH}")

model = YOLO(MODEL_PATH)

# Đọc video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Không thể mở video: {VIDEO_PATH}")

# Lấy thông số video
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 25

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Tạo video output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

if not out.isOpened():
    cap.release()
    raise RuntimeError(f"Không thể tạo file output: {OUTPUT_PATH}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect
    results = model(frame)

    # Vẽ kết quả
    annotated_frame = results[0].plot()

    # Ghi video
    out.write(annotated_frame)

    # Hiển thị
    cv2.imshow("Video Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == 27:  # nhấn ESC để thoát
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Đã lưu video output tại: {OUTPUT_PATH}")