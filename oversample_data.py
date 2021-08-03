import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import pickle

from sklearn.preprocessing import LabelEncoder


def load_data(d):
    
    training_videos = [os.path.join(d,path) for path in os.listdir(d)]
    y = []
    frames = []
    for video_path in training_videos:
        if ".mov" in video_path:
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
    
    return X,y
    


with open("images.pickle", "rb") as f:
    X,y = pickle.load(f)


# results = np.bincount(y)
# total = sum(results)
# print(total)
# classes = ["Start","Release","Loading","Cocking","Accleration","Contact","Deceleration","Finish"]
# print("class percentage:\n")
# for i,class_ in enumerate(results):
#   print(f"{classes[i]}: {class_/total*100:.2f}%")

# unique, counts = np.unique(y, return_counts=True)
# print(dict(zip(unique, counts)))

# plt.bar(unique,counts)
# plt.show()
print(X.shape)
print(y.shape)
X = X.reshape(1855, 224 * 224 * 3)
oversample = SMOTE()
new_X, new_y = oversample.fit_resample(X, y)

new_X = new_X.reshape(new_X.shape[0],224,224,3)
print(new_X.shape)
print(new_y.shape)

results = np.bincount(new_y)
total = sum(results)
print(total)
classes = ["Start","Release","Loading","Cocking","Accleration","Contact","Deceleration","Finish"]
print("class percentage:\n")
for i,class_ in enumerate(results):
  print(f"{classes[i]}: {class_/total*100:.2f}%")
  
  
  
unique, counts = np.unique(new_y, return_counts=True)
print(dict(zip(unique, counts)))

plt.bar(unique,counts)
plt.show()