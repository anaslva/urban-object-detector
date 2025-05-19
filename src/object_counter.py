import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

video_path = "src/videos/39031.avi"  
frame_limit = 100  
os.makedirs("src/frames", exist_ok=True)

cap = cv2.VideoCapture(video_path)
success, frame = cap.read()
frame_count = 0

print("....extraindo frames do vídeo....")

while success and frame_count < frame_limit:
    frame_file = f"src/frames/frame_{frame_count:04d}.jpg"
    cv2.imwrite(frame_file, frame)
    success, frame = cap.read()
    frame_count += 1
    
print(f"!...{frame_count} frames salvos...!")

#modelo 

model = YOLO("yolov5su.pt")

total_count = {}

def update_count(detected_labels):
    for label in detected_labels:
        total_count[label] = total_count.get(label, 0) + 1
        

print("...processando frames com YOLO...")

for i in range(frame_count):
    frame_path = f"src/frames/frame_{i:04d}.jpg"
    results = model(frame_path, verbose=False)[0]
    labels = [model.names[int(cls)] for cls in results.boxes.cls.tolist()]
    update_count(labels)

df = pd.DataFrame(total_count.items(), columns=["Class", "Quantity"])
urban_classes = ["car", "truck", "bus", "motorbike", "bicycle", "person"]
df = df[df["Class"].isin(urban_classes)]
df.sort_values("Quantity", ascending=False, inplace=True)
df.to_csv("final_count.csv", index=False)

print("\ncontagem final por classe:")
print(df)

# gráfico

plt.figure(figsize=(8, 5))
plt.bar(df["Class"], df["Quantity"], color="skyblue")
plt.title("Detected objects count (YOLOv5 - UA-DETRAC)")
plt.ylabel("Quantity")
plt.xlabel("Class")
plt.grid(True)
plt.tight_layout()
plt.show()