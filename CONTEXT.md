# Gesture Assistant Domain Model

The Gesture Assistant translates webcam-captured hand gestures into operating-system media controls and browser automations using real-time machine learning.

## Language

**Gesture**:
A classified static hand posture represented by 21 normalized 3D hand landmarks and a handedness index.
_Avoid_: Pose, sign, motion, hand shape

**Gesture Buffer**:
A fixed-length sliding window (deque) of recent consecutive gesture classifications used to verify pose stability before triggering an action.
_Avoid_: Frame queue, history, cache

**Action**:
A discrete operating-system automation command (playback toggle, volume shift, track navigation, or video seek) executed when a gesture is sustained past the buffer threshold.
_Avoid_: Command, event, task, macro

**Cooldown**:
A duration of time following an action trigger during which subsequent executions of that action category are suppressed to prevent accidental repeats.
_Avoid_: Delay, timeout, sleep, debounce period

**Media System Target**:
The platform-specific backend or desktop application designated to receive actions (e.g., Spotify desktop application, macOS Universal media keys, Google Chrome YouTube tab, or Windows Universal media keys).
_Avoid_: Player, output device, system mode
