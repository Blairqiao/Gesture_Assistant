# Mac Gesture Assistant

[![macOS](https://img.shields.io/badge/OS-macOS-black?style=for-the-badge&logo=apple)](https://www.apple.com/macos/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks_API-0097A7?style=for-the-badge&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Random_Forest-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

**Mac Gesture Assistant** is a real-time computer vision and machine learning tool for macOS that translates hand gestures captured via webcam into system actions and media controls. Built using **Google MediaPipe**, **OpenCV**, and **Scikit-Learn**, it seamlessly controls Spotify playback and system output volume using native macOS AppleScript integration (`osascript`).

---

## Features

- **Real-Time Hand Tracking**: Powered by MediaPipe Tasks API (`HandLandmarker`) tracking 21 3D hand landmarks per hand with live-stream video feed.
- **Custom Gesture Classifier**: Trained Random Forest model classifying custom hand posture vectors.
- **Robust Feature Normalization**:
  - **Origin Invariance**: Wrist landmark $(P_0)$ set as the origin $(0,0,0)$.
  - **Scale Invariance**: Coordinates scaled by the Euclidean distance between wrist $(P_0)$ and middle finger MCP $(P_9)$.
  - **Handedness Aware**: Incorporates left/right hand classification into the feature representation.
- **Gesture Debouncing & Smoothing**: Deque-based buffer ($15$ consecutive matching frames) with dynamic cooldowns ($0.3\text{s}$ for volume adjustments, $1.5\text{s}$ for track control) to eliminate jitter and unintended triggers.
- **Native macOS Control**: Uses AppleScript (`osascript`) for direct, low-overhead system and Spotify automation.
- **Interactive Visualizer**: Built-in visual debugging tool overlaying hand landmark skeletons and real-time prediction overlays onto webcam feeds.
- **Complete Pipeline**: Includes data collection utilities, visual debugging scripts, and model training code.

---

## Supported Gestures & Actions

| Gesture | Label | Action Triggered | Target Application / System |
| :--- | :--- | :--- | :--- |
| 🖐️ **Palm Up** | `palm up` | Play / Pause | Spotify |
| 👍 **Thumbs Up** | `thumbs up` | Increase Volume (+10%) | macOS System Volume |
| 👎 **Thumbs Down** | `thumbs down` | Decrease Volume (-10%) | macOS System Volume |
| 👈 **Point Left** | `point left` | Previous Track | Spotify |
| 👉 **Point Right** | `point right` | Next Track | Spotify |

---

## Repository Structure

```
Mac_Gesture_Assistant/
├── gesture_assistant.py        # Main application running hand tracking & background triggers
├── gesture_visualizer.py       # Debug visualizer displaying webcam feed & gesture predictions
│
├── Models/
│   ├── gesture_model.pkl       # Serialized Random Forest classifier
│   └── hand_landmarker.task    # MediaPipe hand landmarker vision task model
│
├── Data_Collection_Training/
│   ├── train.py                # Script to train Random Forest classifier on annotated dataset
│   ├── hand_feature_extraction_visualizer.py # Utility to preview landmark extraction
│   └── annotated_data.csv      # Formatted CSV dataset containing hand feature vectors
│
├── README.md                   # Project documentation
└── .gitignore                  # Git ignore specifications
```

---

## Getting Started

### Prerequisites

- **Operating System**: macOS (AppleScript integration requires macOS)
- **Python**: Version 3.9 or higher
- **Webcam**: Built-in FaceTime HD camera or USB webcam

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/YourUsername/Mac_Gesture_Assistant.git
   cd Mac_Gesture_Assistant
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install opencv-python mediapipe scikit-learn joblib numpy pandas
   ```

4. **macOS System Permissions**:
   - Ensure Terminal / your IDE has **Camera** permissions enabled (*System Settings > Privacy & Security > Camera*).
   - If prompted when controlling Spotify or system volume, grant automation permissions for `osascript`.

---

## Usage

### 1. Run the Main Assistant
To start controlling Spotify and System Volume in the background:
```bash
python gesture_assistant.py
```
*Press `Ctrl+C` in the terminal to exit.*

### 2. Run the Gesture Visualizer
To view live camera output with rendered hand skeletons and predicted gesture labels:
```bash
python gesture_visualizer.py
```
*Press `q` while focused on the video window to exit.*

### 3. Re-train the Gesture Model
To retrain the Random Forest model using `Data_Collection_Training/annotated_data.csv`:
```bash
python Data_Collection_Training/train.py
```
The newly trained model will be exported as `gesture_model.pkl`.

---

## Technical Details

### Feature Vector Normalization
Hand landmark coordinates extracted by MediaPipe are 3D point positions relative to image dimensions. To ensure gesture recognition is invariant to hand position and distance from camera:

1. **Shift to Wrist Origin**:
   $$\mathbf{P}'_i = \mathbf{P}_i - \mathbf{P}_{\text{wrist}}$$
2. **Scale Invariance**:
   $$d = \|\mathbf{P}'_{\text{middle\_mcp}}\|_2$$
   $$\mathbf{P}''_i = \frac{\mathbf{P}'_i}{d}$$
3. **Feature Concatenation**:
   The final feature vector consists of $1$ handedness index followed by $21 \times 3 = 63$ normalized $(x, y, z)$ landmark coordinates ($64$ features total).

### Model Architecture
- **Classifier**: `RandomForestClassifier` (100 estimators, max depth 15)
- **Evaluation Metric**: Stratified accuracy evaluation on unseen test split (~20%).

---

## Tech Stack

- **Computer Vision**: OpenCV (`cv2`), MediaPipe (`mediapipe.tasks.python.vision`)
- **Machine Learning**: Scikit-Learn (`sklearn.ensemble.RandomForestClassifier`), Joblib
- **System Integration**: Python `subprocess`, macOS `osascript` (AppleScript)

---

## License

This project is open source under the MIT License.