## Project Overview

The project is 
This repository contains the semantic segmentation implementation in pytorch on cityscapes dataset. 

### Dataset

Data can be downloaded from the offical website of cityscapes (https://www.cityscapes-dataset.com/). For training images download leftImg8bit zip file and for corresponding labels download gtFine zip.

There are 20 classes in the dataset.


### Requirements
Pytorch
PIL
cv2
matplotlib
numpy
tqdm

### Training

The Unet architecture is used for training, which consits of encoder and decoder.

### Loss

Cross Entropy and dice loss are available but for training, primarily cross entropy is used. Dataset is highly imbalance, hence classes are assigned weights which are used to calculate cross entropy.

