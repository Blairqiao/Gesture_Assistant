# Mac & Windows Gesture Assistant

[![macOS](https://img.shields.io/badge/OS-macOS-black?style=for-the-badge&logo=apple)](https://www.apple.com/macos/)
[![Windows](https://img.shields.io/badge/OS-Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)](https://www.microsoft.com/windows)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks_API-0097A7?style=for-the-badge&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Random_Forest-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

**Gesture Assistant** is a real-time computer vision and machine learning tool for **macOS** and **Windows** that translates hand gestures captured via webcam into system actions and media controls. Built using **Google MediaPipe**, **OpenCV**, and **Scikit-Learn**, it seamlessly controls media playback and system volume using native macOS AppleScript (`osascript`) / Quartz integration and Windows Win32 API media keys (`ctypes`).

---

## Features

- **Real-Time Hand Tracking**: Powered by MediaPipe Tasks API (`HandLandmarker`) tracking 21 3D hand landmarks per hand with live-stream video feed.
- **Custom Gesture Classifier**: Trained Random Forest model classifying custom hand posture vectors.
- **Robust Feature Normalization**:
  - **Origin Invariance**: Wrist landmark $(P_0)$ set as the origin $(0,0,0)$.
  - **Scale Invariance**: Coordinates scaled by the Euclidean distance between wrist $(P_0)$ and middle finger MCP $(P_9)$.
  - **Handedness Aware**: Incorporates left/right hand classification into the feature representation.
- **Gesture Debouncing & Smoothing**: Deque-based buffer ($15$ consecutive matching frames) with dynamic cooldowns ($0.3\text{s}$ for volume adjustments, $1.3\text{s}$ for track control) to eliminate jitter and unintended triggers.
- **Cross-Platform OS Automation**:
  - **macOS (`gesture_assistant.py`)**: Uses AppleScript (`osascript`) & `Quartz` for Spotify, universal macOS media keys, and Chrome YouTube controls with `config.toml` support.
  - **Windows (`gesture_asssistant_windows.py`)**: Uses native Win32 API (`ctypes.windll.user32.keybd_event`) for universal Windows media controls (Spotify, YouTube in Chrome/Edge/Firefox, iTunes, Windows Media Player).
- **Interactive Visualizer**: Built-in visual debugging tool overlaying hand landmark skeletons and real-time prediction overlays onto webcam feeds.
- **Complete Pipeline**: Includes data collection utilities, visual debugging scripts, and model training code.

---

## Supported Gestures & Actions

| Gesture | Label | Action Triggered | Target System |
| :--- | :--- | :--- | :--- |
| 🖐️ **Palm Up** | `palm up` | Play / Pause | Spotify / Universal Media |
| 👍 **Thumbs Up** | `thumbs up` | Increase Volume (+10%) | System Volume |
| 👎 **Thumbs Down** | `thumbs down` | Decrease Volume (-10%) | System Volume |
| 👈 **Point Left** | `point left` | Previous Track / Skip Back | Spotify / Universal Media / YouTube |
| 👉 **Point Right** | `point right` | Next Track / Skip Forward | Spotify / Universal Media / YouTube |

---

## Repository Structure

```
Mac_Gesture_Assistant/
├── gesture_assistant.py             # macOS application with config-driven media automation
├── gesture_assistant_windows.py    # Windows standalone application with Win32 media keys
├── gesture_visualizer.py            # Debug visualizer displaying webcam feed & predictions
├── config.toml                      # macOS configuration file (system target, cooldowns)
│
├── Models/
│   ├── gesture_model.pkl            # Serialized Random Forest classifier
│   └── hand_landmarker.task         # MediaPipe hand landmarker vision task model
│
├── Data_Collection_Training/
│   ├── train.py                     # Script to train Random Forest classifier on annotated dataset
│   ├── hand_feature_extraction_visualizer.py # Utility to preview landmark extraction
│   └── annotated_data.csv           # Formatted CSV dataset containing hand feature vectors
│
├── README.md                        # Project documentation
└── .gitignore                       # Git ignore specifications
```

---

## Getting Started

### Prerequisites

- **Operating System**: macOS or Windows 10/11
- **Python**: Version **3.12** *(MediaPipe Tasks API is currently not compatible with Python 3.13+)*
- **Webcam**: Built-in FaceTime HD / laptop camera or external USB webcam

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Blairqiao/Mac_Gesture_Assistant.git
   cd Mac_Gesture_Assistant
   ```

2. **Create and Activate a Virtual Environment**:
   - **macOS / Linux**:
     ```bash
     python3.12 -m venv venv
     source venv/bin/activate
     ```
   - **Windows**:
     ```cmd
     py -3.12 -m venv venv
     venv\Scripts\activate
     ```

3. **Install Dependencies**:
   ```bash
   pip install opencv-python mediapipe scikit-learn joblib numpy pandas
   ```
   *(On macOS, `pyobjc-framework-Quartz` is also used by `gesture_assistant.py`)*
   *(On Windows, `keyboard` is required)*

4. **System Permissions**:
   - **macOS**: Ensure Terminal / your IDE has **Camera** permissions enabled (*System Settings > Privacy & Security > Camera*), as well as Automation permissions if prompted.
   - **Windows**: Ensure Camera privacy settings allow desktop apps to access your camera (*Settings > Privacy & security > Camera*).

---

## Usage

### 1. Run the Assistant

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
The newly trained model will be exported to `Models/gesture_model.pkl`.

---

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

---

## Tech Stack

- **Computer Vision**: OpenCV (`cv2`), MediaPipe (`mediapipe.tasks.python.vision`)
- **Machine Learning**: Scikit-Learn (`sklearn.ensemble.RandomForestClassifier`), Joblib
- **macOS Integration**: Python `subprocess`, AppleScript (`osascript`), `Quartz`
- **Windows Integration**: Win32 API (`ctypes.windll.user32.keybd_event`)

---

## License

This project is open source under the MIT License.