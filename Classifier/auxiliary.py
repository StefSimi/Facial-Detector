import os
import cv2
import random

# Paths
image_folder = "antrenare/dad"  # Path to the folder containing dad's images
annotation_file = "antrenare/dad_annotations.txt"  # Path to the dad annotations file
negative_output_folder = "DD/negative_samples_4000_128_iou"  # Folder to save the negative samples

# Ensure output folder exists
os.makedirs(negative_output_folder, exist_ok=True)

def load_annotations(annotation_file):
    annotations = {}
    with open(annotation_file, "r") as file:
        for line in file:
            parts = line.strip().split()
            image_name = parts[0]
            x1, y1, x2, y2 = map(int, parts[1:5])
            label = parts[5]
            if image_name not in annotations:
                annotations[image_name] = []
            annotations[image_name].append((x1, y1, x2, y2, label))
    return annotations

def is_negative_region(x1, y1, x2, y2, face_boxes):
    """Check if a region does not overlap with any face box."""
    for fx1, fy1, fx2, fy2, _ in face_boxes:
        # Calculate overlap (Intersection Over Union, IoU)
        ix1 = max(x1, fx1)
        iy1 = max(y1, fy1)
        ix2 = min(x2, fx2)
        iy2 = min(y2, fy2)
        intersection_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        box_area = (x2 - x1) * (y2 - y1)
        face_area = (fx2 - fx1) * (fy2 - fy1)
        union_area = box_area + face_area - intersection_area
        if intersection_area / union_area > 0:  # Adjust IoU threshold if needed
            return False
    return True

def generate_negative_sample(image_folder, annotation_file, output_folder, crop_size=(128, 128)):
    annotations = load_annotations(annotation_file)
    images = os.listdir(image_folder)
    negative_id = 0

    for img_name in images:
        img_path = os.path.join(image_folder, img_name)
        if img_name not in annotations:
            continue

        face_boxes = annotations[img_name]
        image = cv2.imread(img_path)
        if image is None:
            continue

        height, width, _ = image.shape
        attempts = 0
        while attempts < 500:
            x1 = random.randint(0, width - crop_size[0])
            y1 = random.randint(0, height - crop_size[1])
            x2 = x1 + crop_size[0]
            y2 = y1 + crop_size[1]

            if is_negative_region(x1, y1, x2, y2, face_boxes):
                crop = image[y1:y2, x1:x2]
                output_path = os.path.join(output_folder, f"neg_0{negative_id:03d}.jpg")
                cv2.imwrite(output_path, crop)
                negative_id += 1
                break

            attempts += 1

        print(f"Generated 1 negative sample for {img_name}.")


generate_negative_sample(image_folder, annotation_file, negative_output_folder)
