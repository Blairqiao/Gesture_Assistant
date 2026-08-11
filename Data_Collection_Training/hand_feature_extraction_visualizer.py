import cv2
import time
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

def result_callback(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

# This method is taken from Google's MediaPipe code example
# https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#scrollTo=_JVO3rvPD4RN&uniqifier=1
def draw_landmarks_on_image(bgr_image, detection_result):
    if not detection_result or not detection_result.hand_landmarks:
        return bgr_image

    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(bgr_image)

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

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

        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

def main():
    print("Starting visualizer... Press 'q' in the video window to quit.")
    
    base_options = python.BaseOptions(model_asset_path='/Users/blair/Desktop/Projects/Mac_Gesture_Assistant/Data_Collection/hand_landmarker.task')
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
        
        annotated_frame = draw_landmarks_on_image(frame, latest_result)
        
        cv2.imshow('Mac Gesture Assistant', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()