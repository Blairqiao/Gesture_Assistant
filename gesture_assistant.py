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
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    
    x_min, x_max = min(x_coordinates), max(x_coordinates)
    y_min, y_max = min(y_coordinates), max(y_coordinates)
    
    width = x_max - x_min
    height = y_max - y_min
    area = width * height
    
    return area
    

def main():
    print("Starting visualizer... Press 'q' in the video window to quit.")
    
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

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        timestamp_ms = int(time.time() * 1000)
        detector.detect_async(mp_image, timestamp_ms)
        
        # terminal break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


