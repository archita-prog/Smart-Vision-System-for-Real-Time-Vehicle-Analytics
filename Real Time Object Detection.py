#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pip', 'install ultralytics')



# In[2]:


import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO


# In[3]:


model = YOLO('yolov8s.pt')


# In[4]:


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :
        colorsBGR = [x,y]
        print(colorsBGR)
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

CAP = cv2.VideoCapture('vidyolov8.mp4')


# In[6]:


import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO('yolov8n.pt')

# Load class names from coco.txt
with open("coco.txt", "r") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]

# Video setup
video_path = r"C:\Users\Archita Shrivastava\Downloads\vidyolov8\vidyolov8.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run detection
    results = model.predict(frame, verbose=False)
    
    # Process detections
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        
        for box, cls_id, conf in zip(boxes, class_ids, confidences):
            x1, y1, x2, y2 = map(int, box)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Get class name from your coco.txt
            class_name = CLASS_NAMES[int(cls_id)]
            label = f"{class_name} {conf:.2f}"
            
            # Text background for better visibility
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
            
            # Put class name text
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Display
    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()


# In[8]:


get_ipython().system('pip install motmetrics')


# In[10]:


# After tracking a video sequence:
acc = mm.MOTAccumulator(auto_id=True)

for frame_idx, tracks in enumerate(all_tracks):
    gt_ids = [...]  # Ground truth IDs for this frame
    tracker_ids = [t.track_id for t in tracks]
    dist_matrix = compute_distances(gt_boxes, tracked_boxes)  # Your distance metric
    
    acc.update(gt_ids, tracker_ids, dist_matrix)

# Generate final report
summary = mm.metrics.compute(acc, metrics=['mota', 'idf1'])
print(f"MOTA: {summary['mota']:.2f}, IDF1: {summary['idf1']:.2f}")


# In[ ]:




