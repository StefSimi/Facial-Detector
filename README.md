## Facial Detector for Dexter's Laboratory

## About

This project implements a facial detection and classification system for characters from the animated show Dexter's Laboratory. The system utilizes machine learning techniques to identify and classify faces in images, distinguishing between Dexter, Dee Dee, Mom, and Dad. It leverages Histogram of Oriented Gradients (HOG) feature extraction and Support Vector Machines (SVM) for classification.

## Features

-Facial Detection: Detects faces in grayscale images using HOG features and a linear SVM classifier.<br>

-Multi-Class Character Recognition: Identifies characters (Dexter, Dee Dee, Mom, Dad) using an SVM model trained with labeled face data.<br>

-Performance Optimization: Implements hard negative mining and non-maximum suppression (NMS) to refine detections.<br>

-Evaluation Metrics: Uses precision-recall analysis and Intersection over Union (IoU) to assess model accuracy.<br>

-Scalability: Applies multi-scale sliding window detection for detecting faces at different scales.<br>


## Implementation

### 1. Data Preprocessing

-Positive face examples are loaded from labeled datasets. <br>

-HOG descriptors are extracted for feature representation. <br>

-The dataset is augmented extended by flipping images. <br>

### 2. Training the SVM Classifier

-LinearSVC from Scikit-Learn is used to train a binary SVM classifier for face detection. <br>

-Hard negative mining is applied by selecting high-confidence false positives from non-face images. <br>

-A multi-class SVM classifier (One-vs-Rest) is implemented for character recognition. <br>


### 3. Face Detection Pipeline

-A sliding window approach with HOG feature extraction at multiple scales is applied. <br>

-Overlapping detections are filtered using Non-Maximum Suppression (NMS). <br>

-Detected faces are classified using the trained multi-class SVM model. <br>

### 4. Character Identification

-Character labels are predicted using the trained SVM model. <br>

-The detected characters, along with their confidence scores, are returned. <br>

### 5. Evaluation

-Precision and recall are measured in order to assess detection accuracy. <br>

-Intersection over Union (IoU) between detected and ground-truth bounding boxes is computed. <br>

-The evaluation results are saved and the detected faces, along with their corresponding bounding boxes, can bi visualized. <br>
