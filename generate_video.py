"""
This script is used to generate a video of a serve with labels predicted by a model
"""
import tensorflow as tf
import cv2
import numpy as np
import os

model = tf.keras.models.load_model("models/myserve_oversample.h5")

label_book = {0:"Start",1:"Release",2:"Loading",3:"Cocking",4:"Accleration",5:"Contact",6:"Deceleration",7:"Finish"}


video_path = "data/back/img-01.mov"

# # convert training videos into training images and labels

frames = []
cap = cv2.VideoCapture("data/back/img-01.mov")
ret = True
while ret:
    ret,img = cap.read()
    if ret:
        resized = cv2.resize(img,(224,224))
        frames.append(resized)

cap.release()
# close all windows
cv2.destroyAllWindows()

fullvideo_X = np.array(frames) / 255.0
print(fullvideo_X.shape)


# {0: 97, 1: 11, 2: 19, 3: 9, 4: 3, 5: 1, 6: 8, 7: 33}



serve01_pred = model.predict(fullvideo_X)
a = np.argmax(serve01_pred,axis=1)
unique, counts = np.unique(a, return_counts=True)
print(dict(zip(unique, counts)))

labels = [label_book[k] for k in a]

# import pickle

# with open("labels.pickle","rb") as f:
#     labels = pickle.load(f)

new_frames = []
cap = cv2.VideoCapture(video_path)
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




out = cv2.VideoWriter('oversampled_serve01.mov',cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),30,(width,height))
 
for i in range(len(new_frames)):
    out.write(new_frames[i])
    
out.release()

