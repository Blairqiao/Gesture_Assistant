import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

latest_result = None

def result_callback(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

def main():
    print("Starting visualizer... Press 'q' in the video window to quit.")
    
    base_options = python.BaseOptions(model_asset_path='/Users/blair/Desktop/Projects/Mac_Gesture_Assistant/Data_Collection/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options, 
        num_hands=2,
        running_mode=vision.RunningMode.VIDEO,
        result_callback=result_callback
    )
    detector = vision.HandLandmarker.create_from_options(options)

    video = cv2.VideoCapture()

if __name__ == "__main__":
    main()