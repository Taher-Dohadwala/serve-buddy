# Serve Buddy
[![Python](https://img.shields.io/badge/Python-3.8-blue?style=flat&logo=Python)](https://www.python.org/downloads/release/python-380/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.5-orange?style=flat&logo=Tensorflow)](https://www.tensorflow.org/api_docs)
[![OpenCV](https://img.shields.io/badge/OpenCV-2.5-green?syle-flat&logo=OpenCV)](https://docs.opencv.org/4.5.2/d6/d00/tutorial_py_root.html)


## Goal

Deploy a real time voice assistant tennis serve coach using edge technology. The AI coach will assess your serve and verbally give feedback to identify incorrect biomechanics.


## Background

Tennis is a sport that requires lots of practice with each type of stroke. However, during most group practices, serves are hardly practiced. If you are looking to practice serves, good news is that you can go out alone and practice. However, to most amateur players, the indication for if you are serving well is whether or not the ball went in and if it looked fast. The reality is that a serve is very technical and without being able to look at yourself you might not be able to catch all the mistakes.

## Purpose

For players wanting to improve their serve, this will allow them to serve and get instant feedback to correct mistakes, without the need of paid coaching.


# Project Breakdown

## 8 Serve Part Classifier
Current reserach shows that a tennis serve can be broken down into 8 different parts. In order for serve buddy to provide accurate coaching, it must know which of the 8 parts of the serve needs correcting. This will allow serve buddy to tailor feedback specific to each part seperatly.

![serve_breakdown](https://user-images.githubusercontent.com/23107070/128399019-f97a7c30-5e64-4e30-91da-25554ea672bb.png)

### Data Collection
I collected 11 videos of me serving, on an Iphone camera at 1080p 30fps. The video behind the server filiming from their back perspective. Currently, asking other tennis players to send videos of them serving to gather a more complete dataset for generalization.

### Data Labeling
For each video I create a smaller clip of each part of the serve. So for every serve I split it into 8 smaller clips for each of the parts. The way I know when to start and stop each clip comes from domain expertise in tennis, as well as, using the figure about as a guideline.

Future work: Create a streamlined process to label data, and get crowd sourced labels from AWS to increase the dataset size.

### Model Architecture:
The model uses MobileNetV2 as it's base, with a few more Dense layers and a new model head with an 8 output softmax.

![8_serve_part_classifier](https://user-images.githubusercontent.com/23107070/128397199-04ce619d-838e-490b-b985-0a59c29e7fb0.png)


**WIP 08/05/21**


## Pose Matching

## Voice Assistant

## Edge Deployment
