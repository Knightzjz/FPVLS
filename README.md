# Face Recognition using Tensorflow [![Build Status][travis-image]][travis]

[travis-image]: http://travis-ci.org/davidsandberg/facenet.svg?branch=master

This is a TensorFlow implementation of the Face Pixelation in Live Video Streaming project of University ML Camp Jeju 2019

## Compatibility
The code is tested using Tensorflow r1.7 under Ubuntu 14.04 with Python 2.7 and Python 3.5. 
## News
| Date     | Update |
|----------|--------|
| 2019-08-02 | Added models trained on Wider-Face and face align model for later recognition. Note that the models uses fixed image standardization|
| 2019-08-02 | Updated to run with previous version of Tensorflow r0.12. Not sure if it runs with older versions of Tensorflow though.|
| 2019-08-01 | Added models trained on Casia-WebFace and VGGFace2. Note that the models uses fixed image standardization.|
| 2019-08-01 | Uploaded the initial implemantion code of FPLV.|

## Pre-trained models
I offerred the pretrained model from some existing models since these pre-trained models are well-tunned and generally may work better in conventional testing cases. You can replace them to yours by running the face_detection_sub_models/detect_face.py /& face_recognition_sub_models/train_test.py.
| Model name      | LFW accuracy | Training dataset | Architecture |
|-----------------|--------------|------------------|-------------|
| [20180408-102900](https://drive.google.com/open?id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz) | 0.9905        | CASIA-WebFace    | [Inception ResNet v1]|
| [20180402-114759](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-) | 0.9965        | VGGFace2      | [Inception ResNet v1]|

NOTE: If you use any of the models, please do not forget to give proper credit to those providing the training dataset as well.

## Inspiration
The code is heavily inspired by the [OpenFace](https://github.com/cmusatyalab/openface) implementation.

## Training data
The [CASIA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) dataset has been used for training. This training set consists of total of 453 453 images over 10 575 identities after face detection. Some performance improvement has been seen if the dataset has been filtered before training. Some more information about how this was done will come later.
The best performing model has been trained on the [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) dataset consisting of ~3.3M faces and ~9000 classes.

## Running training
Currently, the best results are achieved by training the model using softmax loss. Details on how to train a model using softmax loss on the CASIA-WebFace dataset can be found on the page [Classifier training of Inception-ResNet-v1](https://github.com/davidsandberg/facenet/wiki/Classifier-training-of-inception-resnet-v1) and .

## Pre-trained models
### Inception-ResNet-v1 model
A couple of pretrained models are provided. They are trained using softmax loss with the Inception-Resnet-v1 model.
