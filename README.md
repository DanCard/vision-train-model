# Vision Finger Counter

A high-performance, real-time finger counting application built using **MediaPipe's Hand Landmarker (Tasks API)** and OpenCV. This tool accurately detects and counts fingers on up to two hands simultaneously, with advanced logic for robust thumb detection and dynamic visualization.

## Features

- **Real-time Detection:** High-speed tracking powered by Google's MediaPipe.
- **Robust Thumb Logic:** Uses a distance-based geometric check (Thumb Tip to Pinky Base) rather than simple X-axis checks, making it more reliable when your hand is tilted or rotated.
- **Dynamic Visualization:** Dot sizes scale based on hand proximity (depth proxy) for a cleaner, spatial UI.
- **Large Format Display:** Defaults to a high-resolution 2880x1620 window for easy viewing.
- **Automatic Logging:** Finger counts are logged to the console only when they change.
- **Python 3.13+ Compatible:** Built specifically for the modern MediaPipe Tasks API.

## Requirements

- Python 3.9+ (Tested on 3.13)
- Webcam

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DanCard/vision-train-model.git
   cd vision-train-model
   ```

2. **Set up a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Linux/macOS
   # OR
   .\venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script:
```bash
python finger_counter.py
```

### Controls
- **'q'**: Quit the application.
- **Window 'X' Button**: Close and exit.

## Project Structure

- `finger_counter.py`: Main application logic.
- `hand_landmarker.task`: Pre-trained MediaPipe hand landmark model.
- `requirements.txt`: Python package dependencies.

## Technical Details

The script uses normalized landmark coordinates to calculate Euclidean distances. Finger counting for the index through pinky is based on the relative Y-position of the tips vs the MCP (knuckle) joints. The thumb uses the distance between the Tip (4) and the Pinky MCP (17) compared to the IP joint (3) and the Pinky MCP (17).
