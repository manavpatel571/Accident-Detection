import os
from ultralytics import YOLO
import cv2
import random

# Load your YOLO model
model = YOLO(r"F:/Projects/Accident_Detection/models/best(5).pt")  # Path to your YOLO weights

# Input and output video paths
input_video_path = r"F:\Projects\Accident_Detection\Indian_Car_Accident_Narrow_Escapes(720p).mp4"  # Your input video file
output_video_path = "output_video_with_detections.mp4"  # Output video file

# Directory to save accident frames (detected parts only)
accident_folder = r"F:/Projects/Accident_Detection/accident_frames"
os.makedirs(accident_folder, exist_ok=True)

# Generate a random color for each class, with red for the "Accident" class
class_colors = {}
for cls, name in model.names.items():
    if isinstance(name, str) and name.lower() == "accident":  # Ensure name is a string
        class_colors[cls] = (0, 0, 255)  # Red color (BGR format)
    else:
        class_colors[cls] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# Open the input video
cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Perform YOLO detection
    results = model.predict(frame)  # Updated method for inference

    save_accident_frame = False  # Flag to check if "Accident" class is detected

    # Draw detections on the frame
    for result in results:  # Iterate over the Result objects
        for box in result.boxes:  # Iterate over detected boxes
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            conf = box.conf[0]  # Get confidence score
            cls = int(box.cls[0])  # Get class ID
            label = f"{model.names[cls]} {conf:.2f}"

            # Assign color based on class
            color = class_colors[cls]

            # Check if the class is "Accident"
            if model.names[cls].lower() == "accident":
                save_accident_frame = True

                # Crop the detected region (ROI)
                detected_part = frame[y1:y2, x1:x2]  # Extract region of interest (ROI)

                # Save the cropped part as a separate image
                accident_part_path = os.path.join(accident_folder, f"accident_frame_{frame_count:04d}_roi.jpg")
                cv2.imwrite(accident_part_path, detected_part)

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)  # Bounding box thickness
            font_scale = 0.7
            font_thickness = 2
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            text_x, text_y = x1, y1 - 10 if y1 - 10 > 10 else y1 + 10
            cv2.rectangle(frame, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0], text_y + 5), color, -1)
            cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    # Write the frame with detections to the output video
    out.write(frame)

    # Display the frame in a window
    cv2.imshow("Video with Detections", frame)

    # Check for the 'q' key to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Detection complete. Output video saved at: {output_video_path}")
print(f"Accident frames (detected parts) saved in folder: {accident_folder}")
