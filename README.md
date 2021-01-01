# Face Pixelation in Live Video Streaming [![Build Status][travis-image]]

[travis-image]: http://travis-ci.org/davidsandberg/facenet.svg?branch=master

This is a TensorFlow implementation of the<b>Face Pixelation in Live Video Streaming </b>project of <font color="red"><b><I>University ML Camp Jeju 2019</font></b></I>. Some of the code is built on the MTCNN work in [MTCNN-Tensorflow](https://github.com/AITTSMD/MTCNN-Tensorflow) and [MTCNN_face_detection_alignment](https://github.com/kpzhang93/MTCNN_face_detection_alignment).<br>
    
## Compatibility
The code is tested using Tensorflow 1.12.1 under Ubuntu 14.04 with Python 2.7. 
## News
| Date     | Update |
|----------|--------|
| 2019-08-02 | Added models trained on Wider-Face and face align model for later recognition. Note that the models uses fixed image standardization|
| 2019-08-02 | Updated to run with previous version of Tensorflow r0.12. Not sure if it runs with older versions of Tensorflow though.|
| 2019-08-01 | Added models trained on Casia-WebFace and VGGFace2. Note that the models uses fixed image standardization.|
| 2019-08-01 | Uploaded the initial implemantion code of FPLV.|

## Pre-trained models
| Model name      | 
|-----------------|
| [20180408-102900](https://drive.google.com/open?id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz) |
| [20180402-114759](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-)|

I offerred the pretrained model from some existing models since these pre-trained models are well-tunned and generally may work better in conventional testing cases. You can replace them to yours by running the face_detection_sub_models/detect_face.py /& face_recognition_sub_models/train_test.py.

NOTE: If you use any of the models, please do not forget to give proper credit to those providing the training dataset as well.

## Inspiration
The code is heavily inspired by the [OpenFace](https://github.com/cmusatyalab/openface) implementation.

## Training data
The [CASIA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) dataset has been used for training. This training set consists of total of 453 453 images over 10 575 identities after face detection. Some performance improvement has been seen if the dataset has been filtered before training. Some more information about how this was done will come later.
The best performing model has been trained on the [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) dataset consisting of ~3.3M faces and ~9000 classes.

## Running training
Currently, the best results are achieved by training the model using softmax loss. Details on how to train a model using softmax loss on the CASIA-WebFace dataset can be found on the page [Classifier training of Inception-ResNet-v1](https://github.com/davidsandberg/facenet/wiki/Classifier-training-of-inception-resnet-v1).

## Pre-trained models
### Inception-ResNet-v1 model
A couple of pretrained models are provided. They are trained using softmax loss with the Inception-Resnet-v1 model.

## Result

![result1.png](https://github.com/Knightzjz/University-ML-Camp-Jeju-2019/blob/master/models/R1.png)

![result2.png](https://github.com/Knightzjz/University-ML-Camp-Jeju-2019/blob/master/models/R2.png)

## Keep Updated
<b><I>If you want to watch the progress of FPLV, please check [My Github io Page](https://knightzjz.github.io)  or the hooked [My Personal Webpage](https://bluebalwyy.com). </b></>


## License
MIT LICENSE

## References
1. Kaipeng Zhang, Zhanpeng Zhang, Zhifeng Li, Yu Qiao , " Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks," IEEE Signal Processing Letter
2. [MTCNN-MXNET](https://github.com/Seanlinx/mtcnn)
3. [MTCNN-CAFFE](https://github.com/CongWeilin/mtcnn-caffe)
4. [deep-landmark](https://github.com/luoyetx/deep-landmark)

Note: Please cite the paper if you use the code for implementation.

### Citation
    @article{zhou2020personal,
    title={Personal Privacy Protection via Irrelevant Faces Tracking and Pixelation in Video Live Streaming},
    author={Zhou, Jizhe and Pun, Chi-Man},
    journal={IEEE Transactions on Information Forensics and Security},
    volume={16},
    pages={1088--1103},
    year={2020},
    publisher={IEEE}
    }








