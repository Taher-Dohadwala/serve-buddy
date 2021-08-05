"""
This script is used to train a 8-serve part classifier


Input:
(224 x 224 x 3) images

Model must accept a batch of images so the corret shape is (batch_size,224,224,3)

Model:
MobilnetV2 BASE -> Dense layer -> Output layer with dimension 8

"""

import tensorflow as tf
import os
import cv2
import numpy as np

import pickle
from imblearn.over_sampling import SMOTE
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2


SAVE_PATH = "models/myserve_oversample2.h5"

def load_data(d):

  training_videos = [os.path.join(d,path) for path in os.listdir(d) if '.mov' in path]
  print(len(training_videos))

  # convert training videos into training images and labels

  y = []
  frames = []
  for video_path in training_videos:
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
  
  return X,y


def oversample_data(train_X,train_y):
  train_X = train_X.reshape(train_X.shape[0], 224 * 224 * 3)
  oversample = SMOTE()
  train_X, train_y = oversample.fit_resample(train_X, train_y)

  train_X = train_X.reshape(train_X.shape[0],224,224,3)
  
  return train_X,train_y
  
with open("data/training.pickle",'rb') as f:
  X,y = pickle.load(f) 

print("Finished loading data")


# Train test split

# :181 is the entire serve for img-01.mov
train_X = X[181:]
train_y = y[181:]

test_X = X[:181]/255.0
test_y = y[:181]


train_X,train_y = oversample_data(train_X,train_y)

print("Oversampling dataset complete")


def normalize(image,label):
  image = tf.cast(image,tf.float32) / 255.0
  return image,label

train_dataset = tf.data.Dataset.from_tensor_slices((train_X,train_y))
train_dataset

train_dataset = (train_dataset
                 .map(normalize)
                 .shuffle(1674)
                 .batch(32)
)

# load mobilenetv2
base_model = MobileNetV2(weights="imagenet",include_top=False)
base_model.trainable = False


earlystopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_acc', patience=3,
    mode='max', restore_best_weights=True
)

# model architecture 
model = tf.keras.Sequential([
      base_model,
      tf.keras.layers.GlobalAveragePooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(1024,activation='relu'),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Dense(8,activation="softmax",name="output")
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics = ["acc"]
)


history = model.fit(train_dataset,validation_data=(test_X,test_y),epochs=50)

# Save model
model.save(SAVE_PATH)

