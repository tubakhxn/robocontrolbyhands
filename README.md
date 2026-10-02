# Gesture Mech — Agentic AI Robot Control Interface 🤖
# Dev/Creator= tubakhxn

A computer vision + robotics experiment that turns **hand and finger gestures into robot-like actions**.

The current implementation uses MediaPipe hand landmarks to understand finger positions, classify gestures, and drive a stylized on-screen mechanical robot through behaviors such as push-ups, running, scanning, deploying, compiling, and other autonomous states. fileciteturn0file0L74-L110

> **Educational / research project.** The current code controls a simulated on-screen robot rather than a physical robot.

## What It Does

The system follows:

**See → Understand → Decide → Act**

- Detects a hand with MediaPipe
- Tracks 21 hand landmarks
- Extracts finger-extension, pinch, orientation, and hand-center features
- Classifies gesture states
- Maps gestures to robot behaviors
- Smooths robot movement for more natural animation
- Visualizes the hand-to-robot control relationship
- Provides manual and automatic modes
- Displays a real-time futuristic robotics HUD

The gesture features are calculated from hand landmark geometry, including finger distances, pinch distance, hand roll, and hand center. fileciteturn0file0L117-L125

## Gesture Behaviors

The current heuristic classifier supports states including:

| Gesture / State | Robot Behavior |
|---|---|
| Fingers mostly closed | `PUSH-UPS` |
| Index extended | `RUNNING IMPLEMENTATION` |
| Multiple fingers extended | `SCANNING CODEBASE` |
| Index + middle + ring extended | `DEPLOYING BUILD` |
| Index + pinch configuration | `MIGRATING MEMORY` |
| Thumb/pinky configuration | `AUTO MODE` |
| Pinch + middle-finger configuration | `COMPILING` |

These mappings are implemented as rule-based thresholds over the extracted hand features. fileciteturn0file0L129-L146

## Agentic AI Concept

This project is an experiment toward **agentic AI controlling physical systems**.

The current perception-to-action loop is:

1. **Perception** — observe the user's hand
2. **Feature extraction** — convert landmarks into useful signals
3. **Interpretation** — classify the gesture
4. **Decision** — select a robot state/action
5. **Action** — animate the robot
6. **Feedback** — continuously observe the next gesture

The current implementation is not a general-purpose autonomous agent or LLM-powered planner; its decision layer is a predefined gesture/state system.

## Tech Stack

- Python
- OpenCV
- MediaPipe Hand Landmarker
- NumPy
- Real-time gesture classification
- Temporal smoothing
- Procedural robot animation

The code imports OpenCV and NumPy and uses MediaPipe's Hand Landmarker when available. fileciteturn0file0L12-L16 fileciteturn0file0L74-L96

## Installation

```bash
pip install opencv-python mediapipe numpy
```

On the first run, the script automatically downloads the MediaPipe hand-landmarker model if it is not already present. fileciteturn0file0L58-L70

## Run

Default webcam mode:

```bash
python gesture_mech.py
```

If your Python file has a different filename, replace `gesture_mech.py` with that filename.

The script opens the camera, flips the camera image, resizes it, runs hand detection, classifies the gesture, and updates the robot in real time. fileciteturn0file0L668-L731

## Keyboard Controls

| Key | Action |
|---|---|
| `Q` / `ESC` | Exit |
| `H` | Toggle hand landmarks |
| `A` | Toggle automatic mode |
| `S` | Save screenshot |
| `F` | Toggle fullscreen |

These controls are handled directly in the main loop. fileciteturn0file0L831-L846

## Demo Mode

The script includes a built-in demo mode that does not require a camera:

```bash
python gesture_mech.py --demo
```

You can also save a demo frame:

```bash
python gesture_mech.py --demo --snap demo.png
```

The demo uses synthetic hand landmarks and a predefined gesture sequence. fileciteturn0file0L630-L654 fileciteturn0file0L714-L722

## Robot System

The `Robot` class maintains pose state, smooths target movement, and renders the mechanical avatar. It includes dedicated animation behavior for running and push-ups. fileciteturn0file0L235-L255

The robot is rendered procedurally using OpenCV primitives such as polygons, circles, lines, and ellipses. fileciteturn0file0L197-L227

## HUD

The interface includes:

- Agent status
- Robot mode
- FPS
- Finger activity channels
- Link status
- Auto/manual state
- Hand skeleton
- Gesture-to-robot connection strings
- Animated terminal activity
- Futuristic scanline/vignette effects

The HUD and finger-channel visualization are implemented in the rendering layer. fileciteturn0file0L562-L608

## System Architecture

```text
Camera
  ↓
MediaPipe Hand Landmarker
  ↓
21 Hand Landmarks
  ↓
Feature Extraction
  ↓
Gesture Classification
  ↓
Robot State / Action
  ↓
Pose Smoothing
  ↓
Procedural Robot Animation
  ↓
Visual Feedback
  ↺
```

## Physical Robot Extension

The current project stops at a simulated on-screen robot. A future hardware version could replace the animation/action layer with commands for:

- Arduino
- Raspberry Pi
- ESP32
- Servo motors
- Robotic arms
- Humanoid robot platforms

A physical version could extend the pipeline to:

```text
Camera
  ↓
Vision
  ↓
Gesture / Pose Understanding
  ↓
Agent Decision Layer
  ↓
Robot Command
  ↓
Motors / Servos
  ↓
Physical Robot
  ↓
Camera Feedback
```

## Limitations

This is an experimental prototype:

- Gesture recognition is heuristic and rule-based.
- One hand is tracked.
- Gesture meanings are predefined in code.
- The robot is currently simulated on screen.
- Automatic mode uses predefined/randomized behaviors.
- There is no general-purpose LLM agent or autonomous task planner.
- No safety-critical physical robot control is implemented.

## Future Ideas

- Connect gestures to a real robot
- Add two-hand interaction
- Add full-body pose estimation
- Add natural-language commands
- Add an LLM-based planning layer
- Add memory and task planning
- Learn custom user gestures
- Add reinforcement learning for robot behaviors
- Add robot telemetry
- Add closed-loop visual feedback
- Add push-up and exercise counting
- Add dance and movement imitation
- Build a physical AI personal-training robot

## Safety

This project is intended for **educational, experimental, and research purposes**.

Do not connect the current prototype directly to safety-critical machinery. Any physical-robot deployment should include independent safety controls, hardware limits, emergency stops, testing, and human supervision.
