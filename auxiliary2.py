import os
from PIL import Image

# Input and output directories
input_folder = "preprocessed_faces"
output_folder = "processed_faces"

# Create the output folder structure
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each subfolder
for subfolder in os.listdir(input_folder):
    subfolder_path = os.path.join(input_folder, subfolder)
    if os.path.isdir(subfolder_path):
        # Create corresponding output subfolder
        output_subfolder = os.path.join(output_folder, subfolder)
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)

        # Process each image in the subfolder
        for filename in os.listdir(subfolder_path):
            input_image_path = os.path.join(subfolder_path, filename)
            output_image_path = os.path.join(output_subfolder, filename)

            try:
                # Open the image, resize, and convert to greyscale
                with Image.open(input_image_path) as img:
                    img_resized = img.resize((36, 36))
                    img_greyscale = img_resized.convert("L")
                    img_greyscale.save(output_image_path)
            except Exception as e:
                print(f"Error processing {input_image_path}: {e}")

print("Image processing complete!")
