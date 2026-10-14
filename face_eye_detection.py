import os
import sys
import time
import cv2
from detector import EyeStateDetector, AlertnessTracker

# Video source: command line arg, or local satyam.mp4, or webcam (0)
default_video = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'satyam.mp4')
if len(sys.argv) > 1:
    video_source = sys.argv[1]
elif os.path.exists(default_video):
    video_source = default_video
else:
    video_source = 0

# Initialize VideoCapture
if isinstance(video_source, int) or str(video_source).isdigit():
    cap = cv2.VideoCapture(int(video_source), cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(int(video_source))
else:
    cap = cv2.VideoCapture(video_source)

# Initialize Detector & Alertness Duration Tracker
detector = EyeStateDetector()
# Drowsy if eyes closed > 1s, Sleeping if eyes closed > 3s (as required in PDF)
tracker = AlertnessTracker(drowsy_threshold_sec=1.0, sleep_threshold_sec=3.0)

video_fps = cap.get(cv2.CAP_PROP_FPS)
FPS = video_fps if video_fps and video_fps > 0 else 30.0


def process_video():
    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_source}")
        return

    frame_idx = 0
    start_time = time.time()

    print(f"==================================================")
    print(f" Real-Time Alertness Detection (Awake / Drowsy / Sleep)")
    print(f" Source: {video_source}")
    print(f" Thresholds: Drowsy > 1.0s | Sleeping > 3.0s")
    print(f" Press 'q' or 'ESC' in the window to exit.")
    print(f"==================================================\n")

    while cap.isOpened():
        begin = time.time()
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break

        frame_idx += 1
        # Timestamp based on video playback or wall clock
        timestamp = frame_idx / FPS if isinstance(video_source, str) else (time.time() - start_time)

        # Detect face & eyes
        det = detector.detect(frame)
        status, score, closed_duration = tracker.update(det["is_eyes_closed"], timestamp)

        # Visual overlay
        dst = frame.copy()
        if det["face_detected"] and det["bbox"]:
            x, y, w, h = det["bbox"]

            # Color coding per requirement:
            # Green = Awake, Yellow = Drowsy, Red = Sleeping
            if status == "awake":
                box_color = (0, 255, 0)      # Green
            elif status == "drowsy":
                box_color = (0, 255, 255)    # Yellow (BGR)
            else:
                box_color = (0, 0, 255)      # Red

            # Draw face bounding box
            cv2.rectangle(dst, (x, y), (x + w, y + h), box_color, 2)

            # Draw Alertness Header Badge
            label_text = f"{status.upper()} ({int(score)}%)"
            text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            tw, th = text_size
            banner_y1 = max(0, y - th - 12)
            banner_y2 = y
            cv2.rectangle(dst, (x, banner_y1), (x + tw + 14, banner_y2), box_color, -1)
            cv2.putText(dst, label_text, (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Subtitle detail below bounding box
            detail_text = f"Closed: {closed_duration:.1f}s | EAR: {det['ear_score']} | Eyes: {det['eyes_detected_count']}"
            cv2.putText(dst, detail_text, (x, min(dst.shape[0] - 10, y + h + 22)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

            if frame_idx % 15 == 0:
                print(f"[Frame {frame_idx:04d} | {timestamp:5.1f}s] Status: {status.upper():8s} | "
                      f"Score: {score:5.1f} | Closed: {closed_duration:4.1f}s")

        cv2.imshow('Real-Time Alertness Detection', dst)

        elapsed_ms = (time.time() - begin) * 1000
        delay = max(1, int(1000 / FPS - elapsed_ms))

        key = cv2.waitKey(delay) & 0xFF
        if key in (ord('q'), 27):
            print("Playback interrupted by user.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    process_video()
