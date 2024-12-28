import os
import cv2
import random
import threading
import queue
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# ====================== Configuration ====================== #

# Paths Configuration
YOLO_WEIGHTS_PATH = r"F:/Projects/Accident_Detection/best(5).pt"  # Path to your YOLO weights
INPUT_VIDEO_PATH = r"F:/Projects/Accident_Detection/Indian_Car_Accident_Narrow_Escapes(720p).mp4"  # Input video file
OUTPUT_VIDEO_PATH = "output_video_with_detections.mp4"  # Output video file

ACCIDENT_FRAMES_FOLDER = r"F:/Projects/Accident_Detection/accident_frames"
DESCRIPTION_FOLDER = r"F:/Projects/Accident_Detection/description"

# Ensure directories exist
os.makedirs(ACCIDENT_FRAMES_FOLDER, exist_ok=True)
os.makedirs(DESCRIPTION_FOLDER, exist_ok=True)

# Queue for communication between detection and description threads
description_queue = queue.Queue()

# Sentinel object to signal the description thread to terminate
SENTINEL = None

# ====================== Description Thread ====================== #

def description_worker(q, description_folder):
    """
    Worker function for processing accident images and generating descriptions.
    """
    # Load BLIP model and processor inside the thread
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    while True:
        img_path = q.get()
        if img_path is SENTINEL:
            break  # Exit the thread
        
        try:
            print(f"Analyzing {os.path.basename(img_path)}...")
            description, criticality = analyze_accident_image(img_path, processor, blip_model)
            
            # Create a text file with the description and criticality level
            img_filename = os.path.basename(img_path)
            text_filename = os.path.join(description_folder, f"{os.path.splitext(img_filename)[0]}.txt")
            with open(text_filename, 'w') as file:
                file.write(f"Description: {description}\n")
                file.write(f"Criticality: {criticality}\n")
            
            print(f"Description saved: {text_filename}")
            print('-' * 50)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
        finally:
            q.task_done()

def analyze_accident_image(image_path, processor, model):
    """
    Analyze an image using the BLIP model to generate a description and determine criticality.
    """
    # Open image
    raw_image = Image.open(image_path).convert("RGB")

    # Preprocess the image and pass through the model
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)

    # Decode the output to get the description
    description = processor.decode(out[0], skip_special_tokens=True)

    # Determine criticality based on description
    if "accident" in description.lower():
        criticality = "High"
    else:
        criticality = "Normal"

    return description, criticality

# ====================== Detection and Description Integration ====================== #

def main():
    # Initialize YOLO model
    yolo_model = YOLO(YOLO_WEIGHTS_PATH)  # Load YOLO model

    # Generate random colors for each class, with red for the "Accident" class
    class_colors = {}
    for cls, name in yolo_model.names.items():
        if isinstance(name, str) and name.lower() == "accident":
            class_colors[cls] = (0, 0, 255)  # Red color (BGR)
        else:
            class_colors[cls] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    # Open the input video
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {INPUT_VIDEO_PATH}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    frame_count = 0

    # Start the description thread
    desc_thread = threading.Thread(target=description_worker, args=(description_queue, DESCRIPTION_FOLDER), daemon=True)
    desc_thread.start()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Perform YOLO detection
            results = yolo_model.predict(frame, verbose=False)  # Silent prediction

            # Flag to check if "Accident" class is detected
            save_accident_frame = False

            # Draw detections on the frame
            for result in results:  # Iterate over the Result objects
                for box in result.boxes:  # Iterate over detected boxes
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                    conf = box.conf[0].item()  # Get confidence score
                    cls = int(box.cls[0].item())  # Get class ID
                    label = f"{yolo_model.names[cls]} {conf:.2f}"

                    # Assign color based on class
                    color = class_colors.get(cls, (255, 255, 255))  # Default to white if class not found

                    # Check if the class is "Accident"
                    if yolo_model.names[cls].lower() == "accident":
                        save_accident_frame = True

                        # Crop the detected region (ROI)
                        detected_part = frame[y1:y2, x1:x2]  # Extract region of interest (ROI)

                        # Save the cropped part as a separate image
                        accident_part_path = os.path.join(ACCIDENT_FRAMES_FOLDER, f"accident_frame_{frame_count:04d}_roi.jpg")
                        cv2.imwrite(accident_part_path, detected_part)

                        # Enqueue the image path for description
                        description_queue.put(accident_part_path)

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # Bounding box thickness
                    font_scale = 0.5
                    font_thickness = 1
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                    text_x, text_y = x1, y1 - 10 if y1 - 10 > 10 else y1 + 10
                    cv2.rectangle(frame, (text_x, text_y - text_size[1] - 5),
                                  (text_x + text_size[0], text_y + 5), color, -1)
                    cv2.putText(frame, label, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

            # Write the frame with detections to the output video
            out.write(frame)

            # Display the frame in a window
            cv2.imshow("Video with Detections", frame)

            # Check for the 'q' key to quit early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Early termination requested by user.")
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Signal the description thread to terminate
        description_queue.put(SENTINEL)

        # Wait until the description thread has processed all frames
        description_queue.join()
        desc_thread.join()

        print(f"Detection complete. Output video saved at: {OUTPUT_VIDEO_PATH}")
        print(f"Accident frames (detected parts) saved in folder: {ACCIDENT_FRAMES_FOLDER}")
        print(f"Descriptions saved in folder: {DESCRIPTION_FOLDER}")

if __name__ == "__main__":
    main()