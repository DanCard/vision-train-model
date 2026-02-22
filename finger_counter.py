import cv2
import mediapipe as mp
import math
import logging
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def get_distance(p1, p2):
    """Calculate Euclidean distance between two landmarks (normalized coordinates)."""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def main():
    # 1. Create the HandLandmarker object
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # Open Webcam
    cap = cv2.VideoCapture(0)
    
    # Set base resolution to 1280x720 (HD)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = 'MediaPipe Finger Counter 2880x1620'
    
    # Use WINDOW_NORMAL for explicit sizing
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # INCREASE WINDOW SIZE BY 50% (from 1920x1080 to 2880x1620)
    target_w, target_h = 2880, 1620
    cv2.resizeWindow(window_name, target_w, target_h)
    
    # POSITION HIGHER (adjusting for the larger window)
    cv2.moveWindow(window_name, 50, 20)

    print(f"Finger Counter V8 (Large Window & Small Dots) started. Press 'q' or close to quit.")
    logging.info(f"Targeting window size: {target_w}x{target_h}")

    last_logged_count = -1

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # Process the frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = detector.detect(mp_image)

        # Flip for selfie view
        display_frame = cv2.flip(frame, 1)
        h_disp, w_disp, _ = display_frame.shape

        total_fingers = 0

        if detection_result.hand_landmarks:
            for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
                landmarks = hand_landmarks
                fingers = []

                # --- SMALLER DYNAMIC DOT SIZE ---
                # Wrist (0) to middle finger base (9)
                hand_scale = get_distance(landmarks[0], landmarks[9])
                
                # REVISED DOT SCALING: Much smaller coefficients
                # Previously: hand_scale * w_disp * 0.12 (maxed at 30)
                # Now: maxed at 15 for a "cleaner" look.
                dot_radius = int(hand_scale * w_disp * 0.08)
                dot_radius = max(3, min(dot_radius, 15)) 

                # Thumb Logic (Distance-based)
                dist_tip = get_distance(landmarks[4], landmarks[17])
                dist_ip = get_distance(landmarks[3], landmarks[17])
                
                if dist_tip > dist_ip:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # 4 Fingers (Vertical logic)
                for tip_id in [8, 12, 16, 20]:
                    if landmarks[tip_id].y < landmarks[tip_id - 2].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                total_fingers += fingers.count(1)

                # Draw landmarks with SMALLER radius
                for lm in landmarks:
                    cx = int((1.0 - lm.x) * w_disp)
                    cy = int(lm.y * h_disp)
                    # Simple clean circles
                    cv2.circle(display_frame, (cx, cy), dot_radius, (0, 255, 0), cv2.FILLED)
                    cv2.circle(display_frame, (cx, cy), dot_radius, (0, 0, 0), 1)

        # Log count to terminal ONLY when it changes
        if total_fingers != last_logged_count:
            logging.info(f"Fingers detected: {total_fingers}")
            last_logged_count = total_fingers

        # UI Overlay (Scaled for the high resolution window)
        cv2.rectangle(display_frame, (30, 30), (200, 200), (0, 255, 0), cv2.FILLED)
        cv2.putText(display_frame, str(total_fingers), (65, 165), 
                    cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 15)

        cv2.imshow(window_name, display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except Exception:
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
