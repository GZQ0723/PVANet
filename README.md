# PVANet
PVANET: Lightweight Deep Neural Networks for Real-time Object Detection
by Sanghoon Hong, Byungseok Roh, Kye-hyeon Kim, Yeongjae Cheon, Minje Park (Intel Imaging and Camera Technology)

## Introduction

This repository is a fork from py-faster-rcnn and demonstrates the performance of PVANET.

You can refer to py-faster-rcnn README.md and faster-rcnn README.md for more information.

## Desclaimer

Please note that this repository doesn't contain our in-house runtime code used in the published article.

The original py-faster-rcnn is quite slow and there exist lots of inefficient code blocks.
We improved some of them, by 1) replacing the Caffe backend with its latest version (Sep 1, 2016), and 2) porting our implementation of the proposal layer.
However it is still slower than our in-house runtime code due to the image pre-processing code written in Python (+9ms) and some poorly implemented parts in Caffe (+5 ms).
PVANET was trained by our in-house deep learning library, not by this implementation.
There might be a tiny difference in VOC2012 test results, because some hidden parameters in py-faster-rcnn may be set differently with ours.
PVANET-lite (76.3% mAP on VOC2012, 10th place) is originally designed to verify the effectiveness of multi-scale features for object detection, so it only uses Inception and hyper features only. Further improvement may be achievable by adding C.ReLU, residual connections, etc.
Citing PVANET

The BibTeX for EMDNN2016-accepted version will be updated soon

Installation

Clone the Faster R-CNN repository
# Make sure to clone with --recursive
git clone --recursive https://github.com/sanghoon/pva-faster-rcnn.git
We'll call the directory that you cloned Faster R-CNN into FRCN_ROOT. Build the Cython modules

cd $FRCN_ROOT/lib
make
Build Caffe and pycaffe

cd $FRCN_ROOT/caffe-fast-rcnn
# Now follow the Caffe installation instructions here:
#   http://caffe.berkeleyvision.org/installation.html
# For your Makefile.config:
#   Uncomment `WITH_PYTHON_LAYER := 1`

cp Makefile.config.example Makefile.config
make -j8 && make pycaffe
Download PVANET caffemodels

cd $FRCN_ROOT
./models/pvanet/download_models.sh
If it does not work,
Download full/test.model and move it to ./models/pvanet/full/
Download comp/test.model and move it to ./models/pvanet/comp/
(Optional) Download original caffemodels (without merging batch normalization and scale layers)
cd $FRCN_ROOT
./models/pvanet/download_original_models.sh
If it does not work,
Download full/original.model and move it to ./models/pvanet/full/
Download comp/original.model and move it to ./models/pvanet/comp/
(Optional) Download ImageNet pretrained models
cd $FRCN_ROOT
./models/pvanet/download_imagenet_models.sh
If it does not work,
Download imagenet/original.model and move it to ./models/pvanet/imagenet/
Download imagenet/test.model and move it to ./models/pvanet/imagenet/
(Optional) Download PVANET-lite models
cd $FRCN_ROOT
./models/pvanet/download_lite_models.sh
If it does not work,
Download lite/original.model and move it to ./models/pvanet/lite/
Download lite/test.model and move it to ./models/pvanet/lite/
Models

PVANET
./models/pvanet/full/test.pt: For testing-time efficiency, batch normalization (w/ its moving averaged mini-batch statistics) and scale (w/ its trained parameters) layers are merged into the corresponding convolutional layer.
./models/pvanet/full/original.pt: Original network structure.
PVANET (compressed)
./models/pvanet/comp/test.pt: Compressed network w/ merging batch normalization and scale.
./models/pvanet/comp/original.pt: Original compressed network structure.
PVANET (ImageNet pretrained model)
./models/pvanet/imagenet/test.pt: Classification network w/ merging batch normalization and scale.
./models/pvanet/imagenet/original.pt: Original classification network structure.
PVANET-lite
./models/pvanet/lite/test.pt: Compressed network w/ merging batch normalization and scale.
./models/pvanet/lite/original.pt: Original compressed network structure.
How to run the demo

Download PASCAL VOC 2007 and 2012
Follow the instructions in py-faster-rcnn README.md
PVANET+ on PASCAL VOC 2007
cd $FRCN_ROOT
./tools/test_net.py --gpu 0 --def models/pvanet/full/test.pt --net models/pvanet/full/test.model --cfg models/pvanet/cfgs/submit_160715.yml
PVANET+ (compressed)
cd $FRCN_ROOT
./tools/test_net.py --gpu 0 --def models/pvanet/comp/test.pt --net models/pvanet/comp/test.model --cfg models/pvanet/cfgs/submit_160715.yml
(Optional) ImageNet classification
cd $FRCN_ROOT
./caffe-fast-rcnn/build/tools/caffe test -gpu 0 -model models/pvanet/imagenet/test.pt -weights models/pvanet/imagenet/test.model -iterations 1000
(Optional) PVANET-lite
cd $FRCN_ROOT
./tools/test_net.py --gpu 0 --def models/pvanet/lite/test.pt --net models/pvanet/lite/test.model --cfg models/pvanet/cfgs/submit_160715.yml
Expected results

PVANET+: 83.85% mAP
PVANET+ (compressed): 82.90% mAP
ImageNet classification: 68.998% top-1 accuracy, 88.8902% top-5 accuracy, 1.28726 loss
PVANET-lite: 79.10% mAP
