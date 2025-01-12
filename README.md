# Accident Detection Project

This project is designed to detect accidents in a given video using YOLO-based object detection. It processes the video, identifies accident frames, and stores them along with their descriptions in separate folders for further analysis.

## Features
- Detects accidents in video files.
- Saves accident frames in the `accident frame` folder.
- Saves accident descriptions in the `description` folder.

## Directory Structure
```
manavpatel571-Accident-Detection/
├── app2.py
├── main.py
├── app.py
├── training/
│   └── train-yolo11-object-detection-on-custom-dataset.ipynb
├── models/
│   ├── best(5).pt
│   └── best(4).pt
└── requirements.txt
```

## Prerequisites
Ensure you have Python installed on your system. It is recommended to use Python 3.7 or higher.

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/manavpatel571/Accident-Detection.git
   cd Accident-Detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the main script:**
   ```bash
   python main.py
   ```

4. **Specify the path of your video file:**
   - When prompted, input the path of the video you want to process for accident detection.

5. **Output:**
   - Detected accident frames will be saved in the `accident frame` folder.
   - Corresponding descriptions will be stored in the `description` folder.

![Description Image]([https://github.com/manavpatel571/AgriTech/blob/main/static/uploads/Screenshot%20(593).png](https://github.com/manavpatel571/Accident-Detection/blob/master/training/InShot_20250103_160953702.jpg))

## Usage Notes
- Ensure your video file is accessible from the specified path.
- Use high-resolution videos for better detection accuracy.

## Contributing
Feel free to contribute to this project by creating pull requests or submitting issues. Your feedback and improvements are always welcome!
