# Face Mask Detection using YOLOv7
This project implements a face mask detection system using the YOLOv7 (You Only Look Once) model. 
The system can detect whether a person is wearing a face mask or not in real-time using a webcam.
The dataset was annotated with bounding boxes and divided into training, testing, and validation sets to train the model effectively.

## Dataset downloading
I downloaded dataset from the Kaggle.
I converted the below code lines to comments. I used this code to download data from Kaggle. 
I downloaded the dataset in my google drive. It is enough to download it once, we do not need to download it more than once.

"""
!pip install -q kaggle

from google.colab import drive
drive.mount('/content/drive')

! mkdir ~/.kaggle

!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/kaggle.json

! chmod 600 ~/.kaggle/kaggle.json

%cd /content/drive/MyDrive/Dataset

! kaggle datasets download andrewmvd/face-mask-detection
"""
## Introduction
The objective of this project is to build a robust face mask detection system using YOLOv7. 
The model is capable of detecting whether a person is wearing a mask, not wearing a mask, or wearing a mask improperly. 
The application can be used in real-time scenarios, such as monitoring compliance with mask mandates in public spaces.

## Dataset
The dataset used for training the model includes images with annotations and bounding boxes for faces with masks, faces without masks, and faces with improperly worn masks. 
The dataset was pre-processed and divided into three sets:

Training Set: Used for training the model.
Validation Set: Used for tuning model parameters.
Testing Set: Used for evaluating model performance.

## Model Architecture
The project utilizes the YOLOv7 architecture, which is an efficient object detection model known for its speed and accuracy. YOLOv7 was chosen for its ability to perform real-time object detection with high precision, making it suitable for detecting small objects like face masks.

Training Process
The training process involves the following steps:

## Data Preprocessing:

Annotated the dataset with bounding boxes around faces.
Divided the dataset into training, validation, and testing sets.
Model Training:

Loaded the YOLOv7 model pre-trained on the COCO dataset.
Fine-tuned the model using the annotated face mask dataset.
Adjusted hyperparameters such as learning rate, batch size, and number of epochs to optimize performance.
Model Validation:

## Evaluated the model on the validation set after each epoch.
Used validation metrics such as precision, recall, and F1-score to monitor model performance.
Evaluation
The model was evaluated on the test set to determine its accuracy in detecting face masks. Key evaluation metrics include:

Precision: The proportion of true positive detections among all positive detections.
Recall: The proportion of true positive detections among all actual positives.
F1-Score: The harmonic mean of precision and recall.
mAP (mean Average Precision): A standard metric for object detection tasks.
Real-Time Prediction
After training, the model was deployed for real-time prediction using a webcam or video feed. The system can detect and classify faces as:

1)With Mask
2)Without Mask

The real-time predictions are displayed with bounding boxes and labels.

