import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def preprocess_images(root_dir, annotation_files, character_folders, output_size=(64, 64)):
    print("started preprocessing")
    """
    Preprocess images for training:
    - Extract faces using annotations.
    - Generate negative samples for non-face regions.

    Args:
        root_dir: Root directory containing character-specific folders.
        annotation_files: List of annotation files with bounding boxes.
        character_folders: List of folders corresponding to each annotation file.
        output_size: Size to which the face and negative crops will be resized.
    Returns:
        X (numpy array): Processed image data.
        y (numpy array): Corresponding labels.
        le (LabelEncoder): Fitted label encoder for character labels.
    """
    positive_samples = []
    positive_labels = []
    negative_samples = []
    negative_labels = []

    le = LabelEncoder()
    le.fit(["dad", "deedee", "dexter", "mom", "unknown"])  # Ensure consistent label encoding

    # Loop through each character folder and its corresponding annotation file
    for annotation_file, folder in zip(annotation_files, character_folders):
        folder_path = os.path.join(root_dir, folder)

        with open(annotation_file, "r") as f:
            annotations = f.readlines()

        # Group annotations by image
        grouped_annotations = {}
        for line in annotations:
            parts = line.strip().split()
            image_name = parts[0]
            x1, y1, x2, y2 = map(int, parts[1:5])
            label = parts[5]

            if image_name not in grouped_annotations:
                grouped_annotations[image_name] = []
            grouped_annotations[image_name].append((x1, y1, x2, y2, label))

        # Process each image in the corresponding folder
        for image_name, bboxes in grouped_annotations.items():
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Warning: Could not load image {image_path}")
                continue

            # Extract positive samples (faces)
            face_annotations = []
            for x1, y1, x2, y2, label in bboxes:
                if label != "unknown":  # Positive sample
                    face_crop = image[y1:y2, x1:x2]
                    if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:  # Avoid invalid crops
                        face_resized = cv2.resize(face_crop, output_size)
                        positive_samples.append(face_resized)
                        positive_labels.append(label)
                        face_annotations.append((x1, y1, x2, y2))  # Save for negative sampling

            # Generate negative samples (non-face regions)
            negatives = generate_negative_samples(image, face_annotations, num_samples=5, output_size=output_size)
            negative_samples.extend(negatives)
            negative_labels.extend(["unknown"] * len(negatives))

            print(str(image_path)+"processed")

    # Combine positive and negative samples
    X = positive_samples + negative_samples
    y = positive_labels + negative_labels

    # Normalize and encode labels
    X = np.array(X) / 255.0
    y_encoded = le.transform(y)
    print("ended preprocessing")
    return X, y_encoded, le


def generate_negative_samples(image, face_annotations, num_samples=5, output_size=(64, 64)):
    """
    Generate random negative samples from an image that do not overlap with annotated faces.

    Args:
        image: The input image.
        face_annotations: List of annotated face bounding boxes.
        num_samples: Number of negative samples to generate.
        output_size: Size to which the negative samples will be resized.
    Returns:
        negatives (list): List of resized negative samples.
    """
    height, width, _ = image.shape
    negatives = []

    for _ in range(num_samples):
        iterations=0
        while True and iterations<1000:
            # Generate random coordinates
            x1 = np.random.randint(0, width - output_size[0])
            y1 = np.random.randint(0, height - output_size[1])
            x2, y2 = x1 + output_size[0], y1 + output_size[1]

            # Check for overlap with face annotations
            overlap = False
            for fx1, fy1, fx2, fy2 in face_annotations:
                if not (x2 < fx1 or x1 > fx2 or y2 < fy1 or y1 > fy2):  # Overlap condition
                    overlap = True
                    break

            if not overlap:
                negative_crop = image[y1:y2, x1:x2]
                if negative_crop.shape[0] > 0 and negative_crop.shape[1] > 0:  # Valid crop
                    negatives.append(cv2.resize(negative_crop, output_size))
                    break
            iterations+=1

    return negatives


def build_model(input_shape, num_classes):
    """
    Build a simple CNN for face classification.

    Args:
        input_shape: Shape of the input images (height, width, channels).
        num_classes: Number of output classes.
    Returns:
        model (Sequential): Compiled CNN model.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

'''
# Set up paths and load the dataset
root_dir = "antrenare"
annotation_files = [
    "antrenare/dad_annotations.txt",
    "antrenare/deedee_annotations.txt",
    "antrenare/dexter_annotations.txt",
    "antrenare/mom_annotations.txt"
]
character_folders = ["dad", "deedee", "dexter", "mom"]

# Preprocess images and prepare the dataset
X, y_encoded, label_encoder = preprocess_images(root_dir, annotation_files, character_folders)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build and train the model
input_shape = X_train.shape[1:]  # (64, 64, 3)
num_classes = len(label_encoder.classes_)
model = build_model(input_shape, num_classes)

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save("face_classifier.h5")
'''

from tensorflow.keras.models import load_model
model = load_model("face_classifier.h5")

import cv2
import numpy as np
import tensorflow as tf


def sliding_window_with_nms(image, model, label_encoder, window_size=(64, 64), step=32, threshold=0.8,
                            iou_threshold=0.5):
    """
    Detect faces in an image using a sliding window approach and apply non-maximum suppression (NMS).

    Args:
        image: Input image for detection.
        model: Trained classification model.
        label_encoder: LabelEncoder for decoding class predictions.
        window_size: Size of the sliding window (width, height).
        step: Step size for sliding the window.
        threshold: Confidence threshold for predictions.
        iou_threshold: IOU threshold for NMS.
    Returns:
        final_detections: List of filtered detections after NMS.
                         Format: [(x1, y1, x2, y2, label, confidence)].
    """
    height, width, _ = image.shape
    detections = []

    # Sliding window
    for y in range(0, height - window_size[1], step):
        for x in range(0, width - window_size[0], step):
            # Extract window
            window = image[y:y + window_size[1], x:x + window_size[0]]
            if window.shape[:2] != window_size:
                continue  # Skip incomplete windows at the edges

            # Normalize and expand dimensions for model prediction
            window_resized = cv2.resize(window, window_size) / 255.0
            window_resized = np.expand_dims(window_resized, axis=0)

            # Predict class probabilities
            predictions = model.predict(window_resized, verbose=0)
            class_idx = np.argmax(predictions)
            confidence = predictions[0, class_idx]
            print(confidence)
            # Add detection if above the confidence threshold and not "unknown"
            label = label_encoder.classes_[class_idx]
            if confidence > threshold: #and label != "unknown":
                print(f"Detected bounding box: {x}, {y}, {window_size[0]}, {y + window_size[1]}, Class: {label}")
                detections.append((x, y, x + window_size[0], y + window_size[1], label, confidence))

    # Apply NMS
    final_detections = apply_nms(detections, iou_threshold)
    return final_detections


def apply_nms(detections, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression (NMS) to filter overlapping boxes.

    Args:
        detections: List of detections (x1, y1, x2, y2, label, confidence).
        iou_threshold: IOU threshold for merging boxes.
    Returns:
        filtered_detections: List of filtered detections.
    """
    if not detections:
        return []

    # Extract boxes and scores
    boxes = np.array([[x1, y1, x2, y2] for (x1, y1, x2, y2, _, _) in detections])
    scores = np.array([conf for (_, _, _, _, _, conf) in detections])

    # Perform NMS
    indices = tf.image.non_max_suppression(boxes, scores, max_output_size=100, iou_threshold=iou_threshold)

    # Filter detections based on NMS results
    filtered_detections = [detections[i] for i in indices.numpy()]
    return filtered_detections


# Example: Test on an image
def visualize_detections(image_path, detections):
    """
    Visualize detections on an image.

    Args:
        image_path: Path to the image.
        detections: List of detections [(x1, y1, x2, y2, label, confidence)].
    """
    image = cv2.imread(image_path)
    for (x1, y1, x2, y2, label, confidence) in detections:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Detections", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

label_encoder = LabelEncoder()
label_encoder.fit(["dad", "deedee", "dexter", "mom", "unknown"])
# Load the model and test
image_path = "antrenare/dad/0002.jpg"
test_image = cv2.imread(image_path)
final_detections = sliding_window_with_nms(test_image, model, label_encoder)

# Visualize results
visualize_detections(image_path, final_detections)
