import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model("models/taher_serve.h5")

label_book = {1:"Start",2:"Release",3:"Loading",4:"Cocking",5:"Accleration",6:"Contact",7:"Deceleration",8:"Finish"}

video_path = "data/back/img-01.mov"

# # convert training videos into training images and labels

frames = []
cap = cv2.VideoCapture(video_path)
ret = True
while ret:
    ret,img = cap.read()
    if ret:
        resized = cv2.resize(img,(224,224))
        frames.append(resized)

cap.release()
# close all windows
cv2.destroyAllWindows()

X = np.array(frames)
X = X / 255.0


predictions = model.predict(X)

labels = []
for pred in predictions:
    labels.append(label_book[np.argmax(pred)])

print("Finish inference")


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




out = cv2.VideoWriter('myvideo.mp4',cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),30,(width,height))
 
for i in range(len(new_frames)):
    out.write(new_frames[i])
    
out.release()

