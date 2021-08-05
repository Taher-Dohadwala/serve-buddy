"""
This script generates a video of the ideal labels for the serve
"""

import cv2
import os
import numpy as np


label_book = {0:"Start",1:"Release",2:"Loading",3:"Cocking",4:"Accleration",5:"Contact",6:"Deceleration",7:"Finish"}

base = 'img-01'
full_video = "data/back/img-01.mov"
d = "data/back/split"
training_videos = [os.path.join(d,path) for path in os.listdir(d) if base in path]
training_videos = ["data/back/split/img-01-1.mov","data/back/split/img-01-2.mov","data/back/split/img-01-3.mov","data/back/split/img-01-4.mov",
                   "data/back/split/img-01-5.mov","data/back/split/img-01-6.mov","data/back/split/img-01-7.mov","data/back/split/img-01-8.mov"]

y = []
frames = []
for video_path in training_videos:
  if ".mov" in video_path:
    #print(f"Working on {video_path}")
    cap = cv2.VideoCapture(video_path)
    ret = True
    while ret:
      ret,img = cap.read()
      if ret:
        resized = cv2.resize(img,(224,224))
        frames.append(resized)
        y.append(int(video_path.split("-")[2].split(".")[0])-1)


X = np.array(frames)
y = np.array(y)


labels = []
for pred in y:
    labels.append(label_book[pred])

print("Finish inference")

print(labels)

new_frames = []
cap = cv2.VideoCapture(full_video)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

idx = 0
ret = True
while ret:
    ret,img = cap.read()
    if ret:
        new_frames.append(cv2.putText(img,labels[idx],(200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA))
        idx += 1

cap.release()
# close all windows
cv2.destroyAllWindows()


out = cv2.VideoWriter('ideal_serve_01.mp4',cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),30,(width,height))
 
for i in range(len(new_frames)):
    out.write(new_frames[i])
    
out.release()