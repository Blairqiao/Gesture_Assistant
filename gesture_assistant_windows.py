from collections import deque
import joblib
from typing import List
import cv2
import time
import math
import keyboard # type: ignore
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BUFFER_SIZE = 15
CD_PLAY_PAUSE = 1.3
CD_VOLUME = 0.3
CD_NEXT_PREV = 1.3

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

    raw_coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]

    # Normalization
    wrist_x, wrist_y, wrist_z = raw_coords[0]
    shifted_coords = [
        (x - wrist_x, y - wrist_y, z - wrist_z) for (x, y, z) in raw_coords
    ]

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
        keyboard.send("play/pause")
    elif gesture == 2:
        print("volume up")
        for _ in range(5):
            keyboard.send("volume up")
    elif gesture == 3:
        print("volume down")
        for _ in range(5):
            keyboard.send("volume down")
    elif gesture == 4:
        print("previous")
        keyboard.send("previous track")
    elif gesture == 5:
        print("next")
        keyboard.send("next track")


def main():
    print("Starting Windows Universal Gesture Assistant...")
    print(f"Target: Universal Windows Media Controls")
    print(f"Cooldowns - Play/Pause: {CD_PLAY_PAUSE}s, Volume: {CD_VOLUME}s, Next/Prev: {CD_NEXT_PREV}s | Buffer Size: {BUFFER_SIZE}")

    model = joblib.load("Models/gesture_model.pkl")

    base_options = python.BaseOptions(model_asset_path='Models/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options, 
        min_hand_detection_confidence = 0.9,
        num_hands=2,
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=result_callback
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # Start webcam feed
    cap = cv2.VideoCapture(0)

    gesture_buffer = deque(maxlen=BUFFER_SIZE) #type: ignore
    last_action_time = 0
    frame_timestamp_ms = 0
    
    print("Windows Gesture Controller Active. Press Ctrl+C in terminal to stop.")

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

            current_time_ms = int(time.time() * 1000)
            if current_time_ms <= frame_timestamp_ms:
                frame_timestamp_ms += 1
            else:
                frame_timestamp_ms = current_time_ms

            try:
                detector.detect_async(mp_image, frame_timestamp_ms)
            except Exception as e:
                print(f"Warning: detect_async failed on frame ({e}). Skipping frame.")
                continue
            
            gesture = parse_result(latest_result, model)

            if gesture is not None:
                gesture_buffer.append(gesture)
            else:
                gesture_buffer.clear()

            if len(gesture_buffer) == BUFFER_SIZE and len(set(gesture_buffer)) == 1:
                current_gesture = gesture_buffer[0]
                
                if current_gesture == 1:
                    cooldown = CD_PLAY_PAUSE
                elif current_gesture in [2, 3]:
                    cooldown = CD_VOLUME
                elif current_gesture in [4, 5]:
                    cooldown = CD_NEXT_PREV
                else:
                    cooldown = 1.0
                
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
