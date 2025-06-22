# Real-Time Facial Expression Detection

This Python application uses OpenCV and MediaPipe to analyze facial expressions in real-time using your webcam. It identifies expressions such as **HAPPY**, **SAD**, **STRESSED**, **ANGRY**, **SURPRISED**, **SMILING**, and **NEUTRAL**.

## Features

✅ Real-time video capture and face detection  
✅ Facial landmarks detection with MediaPipe FaceMesh  
✅ Expression analysis based on mouth, eyes, and eyebrow positions  
✅ Visual overlays of landmarks and expression labels  

## Requirements

- Python 3.7 or higher
- OpenCV (`cv2`)
- MediaPipe
- NumPy

## Installation

1️⃣ Clone this repository:
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
````
2️⃣ Install the required packages:
```bash
pip install opencv-python mediapipe numpy
````


Usage
Run the application:
```bash
python your_script_name.py
(Replace your_script_name.py with your actual Python file name, e.g. expression_analyzer.py)
````

📝 Controls:

Press q to exit the application.

## How it works

- The webcam feed is processed in real-time
- MediaPipe detects facial landmarks
- Distances between key facial points (mouth, eyes, eyebrows) are measured
- Expressions are classified based on these measurements and displayed on the video

## Example & Reference Images
(Replace the above URL with an actual screenshot from your app if you like.)

![Example Screenshot](images/example1.png)

Another view of side:

![Another Example](images/example2.jpg)

## License
This project is licensed under the MIT License. See the LICENSE file for details.
