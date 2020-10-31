## Project Overview (Semantic segmentation)

The project is to segment the obejects present in the images into categories (classes). Concretely, each pixel in the image is classified to one of the 20 classes present in the data. The task of semantic segmentation is also known as dense prediction task, as compared to image classification where whole image is classified to one of the categories, here each pixel has to be classified. 

### Dataset

Data can be downloaded from the offical website of cityscapes (https://www.cityscapes-dataset.com/). For training images download leftImg8bit zip file and for corresponding labels download gtFine zip.

There are 20 classes in the dataset.

There are 3 text files included in the `dataset/` directory which contain the names of all the images and their respective masks for train, val and test sets. Upon downloading the data as mentioned above, one should place the `gtFine` and `leftImg8bit` folders in this `dataset` directory and change the data directory argument in `main.py` file. 


### Requirements
Pytorch </br>
PIL</br>
cv2</br>
matplotlib</br>
numpy</br>
tqdm</br>

### Training

The Unet architecture is used for training, which consits of encoder and decoder. Network is trained for 200 epochs with batch size of 12. 

### Loss

Both Cross Entropy and dice loss are available but for training, primarily cross entropy is used. Dataset is highly imbalance, hence classes are assigned weights which are used to calculate cross entropy.

### Predictions
The progress of the training, losses and metrics could be seen in the log file present in the `exp_unet3` directory. <br>
The output of the network and its actual mask could also be checked in the `images` directory. 

