import subprocess
import gesture_visualizer
from collections import deque
import joblib
from typing import List
import cv2
import time
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mp_hands = mp.tasks.vision.HandLandmarksConnections

latest_result = None
gestures = ["neutral", "palm up", "thumbs up", "thumbs down", "point left", "point right"]

def result_callback(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

def parse_result(detection_result, model):
    if not detection_result or not detection_result.hand_landmarks:
        return None
    
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    
    if len(hand_landmarks_list) == 1:
        return predict_gesture(handedness_list[0], hand_landmarks_list[0], model)
    elif len(hand_landmarks_list) >= 2:
        area1 = hand_area(hand_landmarks_list[0])
        area2 = hand_area(hand_landmarks_list[1])
        if area1 >= area2:
            return predict_gesture(handedness_list[0], hand_landmarks_list[0], model)
        else:
            return predict_gesture(handedness_list[1], hand_landmarks_list[1], model)
    return None

def normalize_coordinates(handedness, hand_landmarks):

    handedness_val = float(handedness[0].index)

    # Raw 3D coordinates [(x, y, z), ...]
    raw_coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]

    # Normalization Pipeline
    # 1. Wrist origin shift: P'_i = P_i - P_0
    wrist_x, wrist_y, wrist_z = raw_coords[0]
    shifted_coords = [
        (x - wrist_x, y - wrist_y, z - wrist_z) for (x, y, z) in raw_coords
    ]

    # 2. Scale invariance: Euclidean distance between Wrist (0) and Middle Finger Base (9)
    middle_mcp_x, middle_mcp_y, middle_mcp_z = shifted_coords[9]
    dist = math.sqrt(middle_mcp_x**2 + middle_mcp_y**2 + middle_mcp_z**2)
    if dist < 1e-6:
        return []

    scaled_coords = [
        (x / dist, y / dist, z / dist) for (x, y, z) in shifted_coords
    ]

    final_vector: List[float] = []
    for x, y, z in scaled_coords:
        final_vector.extend([x, y, z])

    return [handedness_val] + final_vector

def predict_gesture(handedness, hand_landmarks, model):
    features = normalize_coordinates(handedness, hand_landmarks)
    if not features:
        return 0
    prediction = model.predict([features])[0]
    return int(prediction)

def hand_area(hand_landmarks):
    x_coords = [lm.x for lm in hand_landmarks]
    y_coords = [lm.y for lm in hand_landmarks]
    return (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
    
def trigger_action(gesture):
    if gesture == 0:
        return
    elif gesture == 1:
        print("play/pause")
        subprocess.run(['osascript', '-e', 'tell application "Spotify" to playpause'])
        return
    elif gesture == 2:
        print("volume up")
        subprocess.run(['osascript', '-e', 'set volume output volume ((output volume of (get volume settings)) + 10)'])
        return
    elif gesture == 3:
        print("volume down")
        subprocess.run(['osascript', '-e', 'set volume output volume ((output volume of (get volume settings)) - 10)'])
        return
    elif gesture == 4:
        print("previous")
        subprocess.run(['osascript', '-e', 'tell application "Spotify" to previous track'])
        return
    elif gesture == 5:
        print("next")
        subprocess.run(['osascript', '-e', 'tell application "Spotify" to next track'])
        return
    else:
        return

def main():
    print("Starting gesture assistant...")
    
    model = joblib.load("Models/gesture_model.pkl")

    base_options = python.BaseOptions(model_asset_path='Models/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options, 
        num_hands=2,
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=result_callback
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # Start webcam feed
    cap = cv2.VideoCapture(0)

    gesture_buffer = deque(maxlen=15)
    last_action_time = 0
    
    print("Gesture Controller Active. Press Ctrl+C in terminal to stop.")

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int(time.time() * 1000)
            detector.detect_async(mp_image, timestamp_ms)
            
            gesture = parse_result(latest_result, model)

            if gesture is not None:
                gesture_buffer.append(gesture)
            else:
                gesture_buffer.clear()

            if len(gesture_buffer) == 15 and len(set(gesture_buffer)) == 1:
                current_gesture = gesture_buffer[0]
                cooldown = 0.3 if current_gesture in [2, 3] else 1.5
                
                if time.time() - last_action_time > cooldown:
                    trigger_action(current_gesture)
                    last_action_time = time.time()
                    
                    if current_gesture not in [4, 5]:
                        gesture_buffer.clear()
    except KeyboardInterrupt:
        print("\nStopping gesture assistant...")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


