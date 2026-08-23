import tomllib
import os
import sys
import subprocess
from collections import deque
import joblib
from typing import List
import cv2
import time
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import Quartz


def load_config(config_path="config.toml"):
    default_config = {
        "system": "spotify",
        "cooldowns": {
            "play_pause": 1.3,
            "volume": 0.3,
            "next_prev": 1.3
        },
        "youtube_skip_time": 10,
        "buffer_size": 15
    }
    try:
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
            return config
    except Exception as e:
        print(f"Warning: Could not load {config_path} ({e}). Using default configuration.")
        return default_config


def save_config(config, config_path="config.toml"):
    try:
        lines = []
        if "system" in config:
            lines.append(f'system = "{config["system"]}"')
        if "youtube_skip_time" in config:
            lines.append(f'youtube_skip_time = {config["youtube_skip_time"]}')
        if "buffer_size" in config:
            lines.append(f'buffer_size = {config["buffer_size"]}')
        lines.append("")
        if "cooldowns" in config and isinstance(config["cooldowns"], dict):
            lines.append("[cooldowns]")
            for k, v in config["cooldowns"].items():
                lines.append(f"{k} = {v}")
            lines.append("")

        with open(config_path, "w") as f:
            f.write("\n".join(lines))
        print(f"Successfully saved configuration to {config_path}")
    except Exception as e:
        print(f"Error saving {config_path}: {e}")


def run_setup(config_path="config.toml"):
    print("\n==========================================")
    print("      Mac Gesture Assistant Setup         ")
    print("==========================================")
    
    config = load_config(config_path)
    current_system = config.get("system", "spotify")
    print(f"Current System Target: {current_system}\n")
    
    print("Select a media system target:")
    print("  1) spotify  - Control Spotify desktop app")
    print("  2) music    - Universal macOS media keys (any music player)")
    print("  3) youtube  - YouTube playback in Google Chrome (REQUIRES CHROME PERMISSIONS)")
    
    choice = input("\nEnter choice (1-3) or system name [default: spotify]: ").strip().lower()
    
    system_map = {
        "1": "spotify",
        "2": "music",
        "3": "youtube",
        "spotify": "spotify",
        "music": "music",
        "youtube": "youtube"
    }
    
    selected_system = system_map.get(choice, "spotify")
    config["system"] = selected_system
    
    save_config(config, config_path)
    
    print(f"\n[+] Target system updated to '{selected_system}'.")
    print("[i] Note: Advanced settings (cooldowns, youtube_skip_time, etc.) can be directly modified in 'config.toml'.\n")
    
    start_now = input("Would you like to start the gesture assistant now? (Y/n): ").strip().lower()
    if start_now in ["n", "no"]:
        print("Exiting setup.")
        sys.exit(0)


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


def send_media_key(key_type):
    ev_down = Quartz.NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_( #type: ignore
        14,     
        (0, 0),
        0xa00,  
        0, 0, 0,
        8, 
        (key_type << 16) | ((0xa << 8) | 0xa00), 
        -1
    )
    Quartz.CGEventPost(0, ev_down.CGEvent()) #type: ignore

    ev_up = Quartz.NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_( #type: ignore
        14, 
        (0, 0),
        0xb00, 
        0, 0, 0, 
        8, 
        (key_type << 16) | ((0xb << 8) | 0xb00), 
        -1
    )
    Quartz.CGEventPost(0, ev_up.CGEvent()) #type: ignore


def trigger_action(gesture, config=None):
    system = config.get("system", "spotify") if config else "spotify"
    if gesture == 0:
        return
    elif gesture == 1:
        print(f"play/pause ({system})")
        if system == "spotify":
            subprocess.run(['osascript', '-e', 'tell application "Spotify" to playpause'])
        else:
            send_media_key(16)
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
        if system == "spotify":
            subprocess.run(['osascript', '-e', 'tell application "Spotify" to previous track'])
        elif system == "music":
            send_media_key(20)
        else:
            seconds = config.get("youtube_skip_time", 10) if config else 10
            js_payload = f"""
            var v = document.querySelector('video'); 
            if (v && (!v.paused || document.visibilityState === 'visible')) {{
                v.currentTime -= {seconds};
            }}
            """
            
            youtube_prev10 = f'''
            tell application "Google Chrome"
                repeat with w in windows
                    repeat with t in tabs of w
                        if URL of t contains "youtube.com" then
                            execute t javascript "{js_payload}"
                        end if
                    end repeat
                end repeat
            end tell
            '''

            subprocess.run(['osascript', '-e', youtube_prev10])
        return
    elif gesture == 5:
        print("next")
        if system == "spotify":
            subprocess.run(['osascript', '-e', 'tell application "Spotify" to next track'])
        elif system == "music":
            send_media_key(19)
        else:
            seconds = config.get("youtube_skip_time", 10) if config else 10
            js_payload = f"""
            var v = document.querySelector('video'); 
            if (v && (!v.paused || document.visibilityState === 'visible')) {{
                v.currentTime += {seconds};
            }}
            """
            
            youtube_next10 = f'''
            tell application "Google Chrome"
                repeat with w in windows
                    repeat with t in tabs of w
                        if URL of t contains "youtube.com" then
                            execute t javascript "{js_payload}"
                        end if
                    end repeat
                end repeat
            end tell
            '''

            subprocess.run(['osascript', '-e', youtube_next10])
        return
    else:
        return

def main():
    if "--setup" in sys.argv or "-s" in sys.argv:
        run_setup()

    print("Starting gesture assistant...")
    config_path = "config.toml"
    config = load_config(config_path)
    last_config_mtime = os.path.getmtime(config_path) if os.path.exists(config_path) else 0

    system = config.get("system", "spotify") 
    cooldowns = config.get("cooldowns", {}) 
    buffer_size = config.get("buffer_size", 15)
    
    cd_play_pause = cooldowns.get("play_pause", 1.3) #type: ignore
    cd_volume = cooldowns.get("volume", 0.3) #type: ignore
    cd_next_prev = cooldowns.get("next_prev", 1.3) #type: ignore

    print(f"Loaded configuration for system target: {system}")
    print(f"Cooldowns - Play/Pause: {cd_play_pause}s, Volume: {cd_volume}s, Next/Prev: {cd_next_prev}s | Buffer Size: {buffer_size}")

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

    cap = cv2.VideoCapture(0)

    gesture_buffer = deque(maxlen=buffer_size) #type: ignore
    last_action_time = 0
    last_check_time = time.time()
    frame_timestamp_ms = 0
    
    print("Gesture Controller Active. Press Ctrl+C in terminal to stop.")

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    try:
        while cap.isOpened():
            if time.time() - last_check_time > 1.0:
                last_check_time = time.time()
                if os.path.exists(config_path):
                    current_mtime = os.path.getmtime(config_path)
                    if current_mtime > last_config_mtime:
                        config = load_config(config_path)
                        last_config_mtime = current_mtime
                        system = config.get("system", "spotify")
                        cooldowns = config.get("cooldowns", {})
                        new_buffer_size = config.get("buffer_size", 15)
                        
                        if new_buffer_size != buffer_size:
                            buffer_size = new_buffer_size
                            gesture_buffer = deque(gesture_buffer, maxlen=buffer_size) #type: ignore
                            
                        cd_play_pause = cooldowns.get("play_pause", 1.3) #type: ignore
                        cd_volume = cooldowns.get("volume", 0.3) #type: ignore
                        cd_next_prev = cooldowns.get("next_prev", 1.3) #type: ignore
                        print(f"\n[Config Reloaded] System: {system} | Buffer Size: {buffer_size} | Cooldowns: P/P={cd_play_pause}s, Vol={cd_volume}s, N/P={cd_next_prev}s")

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

            if len(gesture_buffer) == buffer_size and len(set(gesture_buffer)) == 1:
                current_gesture = gesture_buffer[0]
                
                if current_gesture == 1:
                    cooldown = cd_play_pause
                elif current_gesture in [2, 3]:
                    cooldown = cd_volume
                elif current_gesture in [4, 5]:
                    cooldown = cd_next_prev
                else:
                    cooldown = 1.0
                
                if time.time() - last_action_time > cooldown:
                    trigger_action(current_gesture, config)
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


