import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load BLIP model and processor (pretrained model for image captioning)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Directory where accident frames (detected parts) are saved
accident_folder = "F:/Projects/Accident_Detection/accident_frames"
description_folder = "F:/Projects/Accident_Detection/description"
os.makedirs(description_folder, exist_ok=True)  # Create description folder if not exists

# Function to analyze images with BLIP model and get a description
def analyze_accident_image(image_path):
    # Open image
    raw_image = Image.open(image_path).convert("RGB")

    # Preprocess the image and pass through the model
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)

    # Decode the output to get the description
    description = processor.decode(out[0], skip_special_tokens=True)

    # Here you can add additional logic for criticality detection
    if "accident" in description.lower():
        criticality = "High"  # Assuming the model detects something related to accidents
    else:
        criticality = "Normal"

    return description, criticality

# Loop through all saved accident frames and analyze them
for img_file in os.listdir(accident_folder):
    img_path = os.path.join(accident_folder, img_file)

    if img_file.endswith('.jpg'):
        print(f"Analyzing {img_file}...")
        description, criticality = analyze_accident_image(img_path)

        # Create a text file with the description and criticality level
        text_filename = os.path.join(description_folder, f"{img_file.replace('.jpg', '.txt')}")
        with open(text_filename, 'w') as file:
            file.write(f"Description: {description}\n")
            file.write(f"Criticality: {criticality}\n")

        # Displaying output for the user
        print(f"Text file saved: {text_filename}")
        print('-' * 50)