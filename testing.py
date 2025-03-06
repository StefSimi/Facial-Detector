import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def sliding_window(image, step, window_size):
    """
    Generate sliding windows for an image.

    Args:
        image: The input image.
        step: Step size for the sliding window.
        window_size: Size of the sliding window (height, width).
    Yields:
        Tuple (x, y, window) where (x, y) is the top-left corner of the window
        and window is the cropped image patch.
    """
    for y in range(0, image.shape[0] - window_size[1] + 1, step):
        for x in range(0, image.shape[1] - window_size[0] + 1, step):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def test_model_on_image(image_path, model_path, label_encoder, window_size=(64, 64), step=32, threshold=0.9):
    """
    Test the trained model on a single image using sliding windows.

    Args:
        image_path: Path to the image to be tested.
        model_path: Path to the trained model.
        label_encoder: Fitted LabelEncoder used during training.
        window_size: Size of the sliding window (height, width).
        step: Step size for the sliding window.
        threshold: Confidence threshold for displaying predictions.
    """
    # Load the image and the model
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    model = load_model(model_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Prepare a copy of the image for visualization
    output_image = image.copy()

    # Normalize the image for model prediction
    image = image / 255.0

    # Sliding window over the image
    for (x, y, window) in sliding_window(image, step, window_size):
        if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
            continue  # Ignore incomplete windows at the edges

        # Expand dimensions for model prediction
        window_input = np.expand_dims(window, axis=0)

        # Predict class probabilities
        preds = model.predict(window_input, verbose=0)
        confidence = np.max(preds)
        label = np.argmax(preds)

        # Check if the confidence exceeds the threshold
        if confidence > threshold:
            # Decode the label
            character = label_encoder.inverse_transform([label])[0]

            # Draw a bounding box and label on the output image
            cv2.rectangle(output_image, (x, y), (x + window_size[0], y + window_size[1]), (0, 255, 0), 2)
            cv2.putText(output_image, f"{character} ({confidence:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the result
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title("Predictions")
    plt.axis("off")
    plt.show()


# Paths and parameters
image_path = "antrenare/dad/0001.jpg"  # Replace with the actual path to 0001.jpg
model_path = "face_classifier.h5"
root_dir = "antrenare"
annotation_files = [
    "antrenare/dad_annotations.txt",
    "antrenare/deedee_annotations.txt",
    "antrenare/dexter_annotations.txt",
    "antrenare/mom_annotations.txt"
]
le = LabelEncoder()
le.fit(["dad", "deedee", "dexter", "mom", "unknown"])

# Test the model
test_model_on_image(image_path, model_path, le, window_size=(64, 64), step=32, threshold=0.99)



