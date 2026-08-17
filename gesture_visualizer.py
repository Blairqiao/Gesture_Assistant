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
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

latest_result = None
gestures = ["neutral", "palm up", "thumbs up", "thumbs down", "point left", "point right"]

def result_callback(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

# This method is taken from Google's MediaPipe code example
# https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#scrollTo=_JVO3rvPD4RN&uniqifier=1
def draw_landmarks_on_image(bgr_image, detection_result, model):
    if not detection_result or not detection_result.hand_landmarks:
        return bgr_image

    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(bgr_image)

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]
        gesture = predict_gesture(handedness, hand_landmarks, model)


        # Draw the hand landmarks using the Tasks API
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        cv2.putText(annotated_image, f"{handedness[0].category_name} - {gestures[gesture]}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

def normalize_coordinates(hand_landmarks):

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

    return final_vector

def predict_gesture(handedness, hand_landmarks, model):
    vector_coordinates = normalize_coordinates(hand_landmarks)
    if not vector_coordinates:
        return 0

    handedness_val = float(handedness[0].index)

    features = [handedness_val] + vector_coordinates
    prediction = model.predict([features])[0]
    
    return int(prediction)
    

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
        
        
        annotated_frame = draw_landmarks_on_image(frame, latest_result, model)
        
        cv2.imshow('Gesture Assistant', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


