# Mac & Windows Gesture Assistant

[![macOS](https://img.shields.io/badge/OS-macOS-black?style=for-the-badge&logo=apple)](https://www.apple.com/macos/)
[![Windows](https://img.shields.io/badge/OS-Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)](https://www.microsoft.com/windows)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks_API-0097A7?style=for-the-badge&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Random_Forest-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

**Gesture Assistant** is a real-time computer vision and machine learning tool for **macOS** and **Windows** that translates hand gestures captured via webcam into system actions and media controls. Built using **Google MediaPipe**, **OpenCV**, and **Scikit-Learn**, it seamlessly controls media playback and system volume using native macOS AppleScript (`osascript`) / Quartz integration and Windows media keys (`keyboard`).

## Table of Contents

- [Features](#features)
- [Supported Gestures & Actions](#supported-gestures--actions)
- [Demo](#demo)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Run the Assistant](#run-the-assistant)
  - [Run the Gesture Visualizer](#run-the-gesture-visualizer)
- [Technical Details](#technical-details)
  - [Feature Vector Normalization](#feature-vector-normalization)
  - [Model Architecture](#model-architecture)
- [Train Your Own Model & Customizing The Assistant](#train-your-own-model--customizing-the-assistant)
  - [Collect Data](#collect-data)
  - [Train the Model](#train-the-model)
  - [Customize the Assistant](#customize-the-assistant)
  - [OSAScript and Python Keyboard Module](#osascript-and-python-keyboard-module)
- [Tech Stack](#tech-stack)
- [License](#license)

## Features

- **Real-Time Hand Tracking**: Powered by MediaPipe Tasks API (`HandLandmarker`) tracking 21 3D hand landmarks per hand with live-stream video feed.
- **Custom Gesture Classifier**: Trained Random Forest model classifying custom hand posture vectors.
- **Custom Gesture Automation**: Uses hand gestures to preform automated tasks such as controlling media playback, system volume, and Youtube media controls.
- **Cross Platform Compatibility**: Works on both macOS and Windows with dedicated scripts for each platform.
- **Gesture Debouncing & Smoothing**: Deque-based buffer ($15$ consecutive matching frames) with dynamic cooldowns ($0.3\text{s}$ for volume adjustments, $1.3\text{s}$ for track control) to eliminate jitter and unintended triggers.
- **Multi-Modal Settings**: Features three distinct modes for controlling media playback: Spotify, YouTube, and Universal media controls (macOS only, the Windows version only has universal controls). 
  - **Spotify**: Only controls the Spotify app, can be used in the background with no impact on other apps. 
  - **Universal**: Controls media playback for the media app in the foreground. Can be used with any media player including Spotify, YouTube(Next Track/Previous Track only works in playlists), Apple Music, etc. 
  - **YouTube**: Same Play/Pause functionality as **Universal** but Next Track/Previous Track is replaced by skip 15 seconds forwards/backwards. The exact time can be configured in `config.toml`.
  
## Supported Gestures & Actions

| Gesture | Label | Action Triggered | Target System |
| :--- | :--- | :--- | :--- |
| 🖐️ **Palm Up** | `palm up` | Play / Pause | Spotify / Universal Media |
| 👍 **Thumbs Up** | `thumbs up` | Increase Volume (+10%) | System Volume |
| 👎 **Thumbs Down** | `thumbs down` | Decrease Volume (-10%) | System Volume |
| 👈 **Point Left** | `point left` | Previous Track / Skip Back 15 Seconds | Spotify / Universal Media / YouTube |
| 👉 **Point Right** | `point right` | Next Track / Skip Forward 15 Seconds | Spotify / Universal Media / YouTube |

## Demo

### Right Hand Tracking
![alt text](/Readme_assets/right_hand_tracking.gif)

### Left Hand Tracking
![alt text](/Readme_assets/left_hand_tracking.gif)

## Repository Structure

```
Gesture_Assistant/
├── gesture_assistant.py             # macOS application with config-driven media automation
├── gesture_assistant_windows.py    # Windows standalone application with Windows media keys
├── gesture_visualizer.py            # Debug visualizer displaying webcam feed & predictions
├── config.toml                      # macOS configuration file (system target, cooldowns)
│
├── Models/
│   ├── gesture_model.pkl            # Serialized Random Forest classifier
│   └── hand_landmarker.task         # MediaPipe hand landmarker vision task model
│
├── Data_Training/
│   ├── train.py                     # Script to train Random Forest classifier on annotated dataset
│   └── annotated_data.csv           # Formatted CSV dataset containing hand feature vectors
│
├── README.md                        # Project documentation
└── .gitignore                       # Git ignore specifications
```

## Getting Started

### Prerequisites

- **Operating System**: macOS or Windows 10/11
- **Python**: Version **3.12** *(MediaPipe Tasks API is currently not compatible with Python 3.13+)*
- **Webcam**: Built-in FaceTime HD / laptop camera or external USB webcam

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Blairqiao/Gesture_Assistant.git
   cd Gesture_Assistant
   ```

2. **Create and Activate a Virtual Environment**:
   - **macOS**:
     ```bash
     python3.12 -m venv venv
     source venv/bin/activate
     ```
   - **Windows**:
     ```cmd
     python -3.12 -m venv venv
     .\venv\Scripts\activate
     ```

3. **Install Dependencies**:
   - **macOS**:
     ```bash
      pip install opencv-python mediapipe scikit-learn joblib numpy pandas pyobjc-framework-Quartz 
     ```
   - **Windows**:
     ```cmd
      pip install opencv-python mediapipe scikit-learn joblib numpy pandas keyboard
     ```

4. **System Permissions**:
   - **macOS**: Ensure Terminal / your IDE has **Camera** permissions enabled (*System Settings > Privacy & Security > Camera*), as well as Automation permissions if prompted.
   - **Windows**: Ensure Camera privacy settings allow desktop apps to access your camera (*Settings > Privacy & security > Camera*).

## Usage

### Run the Assistant

- **On macOS**:
  ```bash
  python gesture_assistant.py
  ```
  *(To switch media targets between Spotify, macOS universal media keys, and Chrome YouTube, run `python gesture_assistant.py --setup`)*

- **On Windows**:
  ```bash
  python gesture_assistant_windows.py
  ```

*Press `Ctrl+C` in the terminal to exit.*

### Run the Gesture Visualizer
To view live camera output with rendered hand skeletons and predicted gesture labels:
```bash
python gesture_visualizer.py
```
*Press `q` while focused on the video window to exit.*

## Technical Details

### Feature Vector Normalization
Hand landmark coordinates extracted by MediaPipe are 3D point positions relative to image dimensions. To ensure gesture recognition is invariant to hand position and distance from the camera:

1. **Shift to Wrist Origin (Translation Invariance)**:
   Subtract the wrist landmark coordinates (P0) from all 21 hand landmarks so that the wrist serves as the coordinate origin (0, 0, 0):
   `P'_i = (x_i - x_wrist, y_i - y_wrist, z_i - z_wrist)`

2. **Scale Invariance**:
   Compute the Euclidean distance between the shifted wrist origin and the middle finger MCP landmark (P9), then divide all shifted coordinates by this reference distance:
   `Distance = sqrt(x_middle_mcp^2 + y_middle_mcp^2 + z_middle_mcp^2)`
   `P''_i = P'_i / Distance`

3. **Feature Concatenation**:
   The final feature vector consists of 1 handedness index (Left/Right) followed by 21 x 3 = 63 normalized (x, y, z) landmark coordinates (64 features total).

### Model Architecture
- **Classifier**: `RandomForestClassifier` (100 estimators, max depth 15)
- **Evaluation Metric**: Stratified accuracy evaluation on unseen test split (~20%).

## Train Your Own Model & Customizing The Assistant
If you want to take this project further to meet your own personal needs/intrests, here is how you can get started:

### Collect Data
1. Record several 30 second videos of yourself(1 for each hand, so 2 videos for each gesture) doing the gestures you want to use for the assistant. Be sure to record some netural poses as well.
2. Use the [Gesture Annotator Tool](https://github.com/Blairqiao/Hand_Gesture_Annotator) to annotate said videos and save the extracted features as a csv.
3. Import the data into this project, replacing 'annotated_data.csv' with your new dataset.

### Train the Model
```bash
python Data_Training/train.py
```
The newly trained model will be exported to the `Models` folder. Be sure to either replace the existing gesture_model.pkl with your own or rename your model and update the `gesture_model.pkl` reference in `gesture_assistant.py` to point to your new model.


### Customize the Assistant
By default, the assistant has 5 gesture slots, with labels 1-5 for each gesture and 0 for no gesture. In the `trigger_gesture()` method, you can modify the scripts associated with each label to perform any action you want. If your model has more than 5 gestures, be sure to add the corresponding mappings within this method. 

### OSAScript and Python Keyboard Module
  - On macOS, you can use Mac's built in apple script([osascript](https://victorscholz.medium.com/what-is-osascript-e48f11b8dec6)) to preform actions.
  - On Windows, you can use the python [keyboard](https://github.com/boppreh/keyboard) module to customize keyboard macros and perform other actions.


## Tech Stack

- **Computer Vision**: OpenCV (`cv2`), MediaPipe (`mediapipe.tasks.python.vision`)
- **Machine Learning**: Scikit-Learn (`sklearn.ensemble.RandomForestClassifier`), Joblib
- **macOS Integration**: Python `subprocess`, AppleScript (`osascript`), `Quartz`
- **Windows Integration**: Python `keyboard` module

## License

This project is open source under the MIT License.

## Developer Profile
**Built by Blair Qiao** | *University of Texas at Austin*
* **GitHub:** [@Blairqiao](https://github.com/Blairqiao)
* **LinkedIn:** [Yanzhe(Blair) Qiao](https://www.linkedin.com/in/yanzhe-qiao-794551413/)
* **Portfolio:** https://blairqiao.com
* **Contact:** yanzhe.qiao.1@gmail.com

