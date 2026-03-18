import os

label_dir = "src/datasets/train/labels"
total = 0

for f in os.listdir(label_dir):
    with open(os.path.join(label_dir, f)) as file:
        total += len(file.readlines())

print("Total objects:", total)