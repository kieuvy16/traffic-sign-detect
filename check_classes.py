from ultralytics import YOLO

model = YOLO("runs/detect/train_fix_60/weights/best.pt")

print("SỐ CLASS:", len(model.names))
print("DANH SÁCH BIỂN BÁO ĐÃ TRAIN:")

for k, v in model.names.items():
    print(f"{k}: {v}")
