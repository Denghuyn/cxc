# CheXNet for Pneumonia Classification

## Overview
This project implements a pre-trained DenseNet-121 model for classifying Chest X-Ray images to detect pneumonia. The model is fine-tuned to fit the specific dataset of chest X-ray images.

## Dataset
About Dataset
#### Context
http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5

![image](https://github.com/hoanganhreaper/chest_xray_image_classification_using_CheXNet/assets/118657851/8b93769c-3e4b-4d1f-ae6f-2f4a223fa2ba)

Figure S6. Illustrative Examples of Chest X-Rays in Patients with Pneumonia, Related to Figure 6
The normal chest X-ray (left panel) depicts clear lungs without any areas of abnormal opacification in the image. Bacterial pneumonia (middle) typically exhibits a focal lobar consolidation, in this case in the right upper lobe (white arrows), whereas viral pneumonia (right) manifests with a more diffuse ‘‘interstitial’’ pattern in both lungs.
http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5

#### Content
The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).
Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.
For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.

#### Acknowledgements
- Data: https://data.mendeley.com/datasets/rscbjbr9sj/2
- License: CC BY 4.0
- Citation: http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5

## Requirements
- Python 3.x
- PyTorch
- TorchVision
- NumPy
- Matplotlib

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/hoanganhreaper/chest_xray_image_classification_using_CheXNet.git
   cd your-repository
2. Install the required packages:
   pip install -r requirements.txt

## Usage
Download the pre-trained DenseNet-121 model weights and place them in the appropriate directory:
   wget https://download.pytorch.org/models/densenet121-a639ec97.pth -P /path/to/weights/

## Model Architecture
The model architecture is based on DenseNet-121. The classifier layer is modified to match the number of classes in the dataset.

## Fine-Tuning
The pre-trained DenseNet-121 model is fine-tuned by:
- Freezing the convolutional layers to prevent them from being updated during training.
- Replacing the final fully connected layer with a new layer that matches the number of classes in the dataset.

## Results
The model achieves the results over 99% in train dataset and 88% in test dataset
