# Lymphoma Subtype Classifier

This repo is a work in progess.  It will be a multi-class classifier looking at three different types of Lymphoma and
classifying their sub-type via its immunostained biopsy image.

##Results

###Optuna


### Tools Used
Pytorch

## Installation

A conda environment has been provided with this project that has all the packages needed by the user to implement this
project.  Simply open up a conda command prompt and enter the command `conda env create --file environment.yml`.  This 
will download all the required packages that the user can utilize via the command `conda activate lymphoma_classifier`.

## Dataset

The dataset consists of immunostained biopsy slides of three different types of Non-Hodgkin's Lymphoma: Chronic
Lymphocytic Leukemia, Follicular Lymphoma, and Mantle Cell Lymphoma.

|     |   Count   |   Percentage  |
| :-- |   :---:   |     :---:     |
| CLL |    113    |      30%      |
| FL  |    139    |      37%      |
| MCL |    122    |      33%      |


The original image dataset can be found here:

[https://www.kaggle.com/andrewmvd/malignant-lymphoma-classification](https://www.kaggle.com/andrewmvd/malignant-lymphoma-classification)

Full credit to the authors for their outstanding work:

Orlov, Nikita & Chen, Wayne & Eckley, David & Macura, Tomasz & Shamir, Lior & Jaffe, Elaine & Goldberg, Ilya. (2010).
Automatic Classification of Lymphoma Images With Transform-Based Global Features. IEEE transactions on information
technology in biomedicine : a publication of the IEEE Engineering in Medicine and Biology Society. 14. 1003-13.
10.1109/TITB.2010.2050695.


## Implementation


### Albumentations

In order to reduce the potential for overfitting, I needed to find a way to increase the size of the training set of
images.  Both PyTorch and 

### Nested K-Fold

### Algorithms

Trained models of seven different algorithm types, ResNet18, ResNet34, ResNet101, MobileNetV2, VGG13, VGG13 with batch
norm, VGG16, and VGG16 with batch norm.  Since the dataset is so small, I utilized pretrained models.

### Training Loop

Because of the richness and size of the images, I did not have the luxury of creating dataloaders to send to the GPU to
be stored in memory.

### Streamlit Web App

In order to launch the web app, use the command `streamlit run /path/to/project/StreamlitApp.py`.
