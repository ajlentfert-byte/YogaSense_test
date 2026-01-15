# Pose_session.py
import cv2
import time
import mediapipe as mp
import os
import numpy as np
import csv
from pose import mountain_pose_score, tree_pose_score, warrior2_pose_score

# -------------------------
# Pose mappings
# -------------------------
POSE_FUNCTIONS = {
    "MOUNTAIN": mountain_pose_score,
    "TREE": tree_pose_score,
    "WARRIOR2": warrior2_pose_score
}

POSE_VIDEOS = {
    "MOUNTAIN": "mountain.MOV",
    "TREE": "tree.MOV",
    "WARRIOR2": "warriorII.MOV"
}

HOLD_TIME = 10.0      # seconds to hold after reaching 90%
TARGET_SCORE = 90.0   # %
LAST_LOG_TIME = 0     # Last time the score was logged

# -------------------------
# Feedback logs
# -------------------------

def generate_feedback(scores):
    feedback = []

    # Knees
    if "front_knee" in scores and scores["front_knee"] < 0.8:
        feedback.append("Front knee not bent enough or too bent")
    if "back_knee" in scores and scores["back_knee"] < 0.8:
        feedback.append("Back leg should be straight")

    # Arms
    if "left_arm" in scores and scores["left_arm"] < 0.8:
        feedback.append("Left arm not level with shoulder")
    if "right_arm" in scores and scores["right_arm"] < 0.8:
        feedback.append("Right arm not level with shoulder")

    # Elbows
    if "left_elbow" in scores and scores["left_elbow"] < 0.8:
        feedback.append("Left elbow not fully extended")
    if "right_elbow" in scores and scores["right_elbow"] < 0.8:
        feedback.append("Right elbow not fully extended")

    # Additional for other poses
    if "standing_leg" in scores and scores["standing_leg"] < 0.8:
        feedback.append("Standing leg should be straight")
    if "lifted_foot" in scores and scores["lifted_foot"] < 0.8:
        feedback.append("Lifted foot not high enough")
    if "hands" in scores and scores["hands"] < 0.8:
        feedback.append("Hands too far apart or too close")
    if "hand_height" in scores and scores["hand_height"] < 0.8:
        feedback.append("Hands not level with shoulders")

    return feedback

# -------------------------
# Circle drawing
# -------------------------

def draw_feedback_circles(frame, lm, scores, threshold=0.8):
    """
    Draws red circles on joints that scored below threshold
    """
    h, w, _ = frame.shape

    def draw_point(landmark, color=(0, 0, 255)):
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(frame, (x, y), 12, color, -1)

    # Warrior II / general mapping
    if "front_knee" in scores and scores["front_knee"] < threshold:
        # Pick front knee landmark (we can infer in pose code or just use left knee for simplicity)
        draw_point(lm[25])  # left knee
    if "back_knee" in scores and scores["back_knee"] < threshold:
        draw_point(lm[26])  # right knee
    if "left_arm" in scores and scores["left_arm"] < threshold:
        draw_point(lm[15])  # left wrist
    if "right_arm" in scores and scores["right_arm"] < threshold:
        draw_point(lm[16])  # right wrist
    if "left_elbow" in scores and scores["left_elbow"] < threshold:
        draw_point(lm[13])  # left elbow
    if "right_elbow" in scores and scores["right_elbow"] < threshold:
        draw_point(lm[14])  # right elbow
    if "hands" in scores and scores["hands"] < threshold:
        draw_point(lm[15])  # left wrist
        draw_point(lm[16])  # right wrist
    if "hand_height" in scores and scores["hand_height"] < threshold:
        draw_point(lm[15])
        draw_point(lm[16])
    if "lifted_foot" in scores and scores["lifted_foot"] < threshold:
        draw_point(lm[27])  # left ankle
        draw_point(lm[28])  # right ankle


# -------------------------
# Main session runner (modified for 0.5s logging)
# -------------------------
def run_pose_session(poses, mode="coach"):
    """
    Run a yoga session with live feedback (coach) or just video+score (video)
    Logs scores to a CSV file every 0.5 seconds after reaching TARGET_SCORE.
    """
    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap_cam = cv2.VideoCapture(0)  # always open camera for scoring

    cv2.namedWindow("Yoga Sense", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Yoga Sense", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Prepare CSV logging
    score_log = []  # (pose_name, score, timestamp)
    csv_file = f"yoga_scores_{int(time.time())}.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Pose", "Score", "Timestamp"])

    for pose_name in poses:
        scorer = POSE_FUNCTIONS[pose_name]
        hold_start = None
        hold_active = False  # True when countdown started
        last_log_time = 0    # Tracks 0.5s logging interval

        # Video setup
        cap_vid = None
        if pose_name in POSE_VIDEOS and os.path.exists(POSE_VIDEOS[pose_name]):
            cap_vid = cv2.VideoCapture(POSE_VIDEOS[pose_name])

        while True:
            ret_cam, frame_cam = cap_cam.read()
            if not ret_cam:
                break

            # Build display first
            display = frame_cam.copy() if mode == "coach" else np.full((480, 640, 3), (211, 177, 211), dtype=np.uint8)

            score = 0
            feedback = []

            if ret_cam:
                rgb = cv2.cvtColor(frame_cam, cv2.COLOR_BGR2RGB)
                result = pose_detector.process(rgb)
                if result.pose_landmarks:
                    lm = result.pose_landmarks.landmark
                    score, _, scores = scorer(lm)

                    if mode == "coach":
                        # Generate textual feedback
                        feedback = generate_feedback(scores)
                        # Draw circles on incorrect joints
                        draw_feedback_circles(display, lm, scores)

            # Overlay pose video (if any)
            if cap_vid:
                ret_vid, frame_vid = cap_vid.read()
                if not ret_vid:
                    cap_vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_vid, frame_vid = cap_vid.read()
                frame_vid = cv2.rotate(frame_vid, cv2.ROTATE_90_COUNTERCLOCKWISE)
                vh, vw = frame_vid.shape[:2]
                scale = display.shape[0] / vh
                new_w = int(vw * scale)
                new_h = display.shape[0]
                frame_vid_resized = cv2.resize(frame_vid, (new_w, new_h))
                display[:, -new_w:] = frame_vid_resized

            # Start hold when score reaches target
            if score >= TARGET_SCORE and not hold_active:
                hold_start = time.time()
                hold_active = True
                last_log_time = time.time()  # log immediately
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                score_log.append((pose_name, score, timestamp))
                with open(csv_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([pose_name, score, timestamp])

            # Hold countdown & periodic logging
            if hold_active:
                elapsed = time.time() - hold_start

                # Log score every 0.5s
                if time.time() - last_log_time >= 0.5:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    score_log.append((pose_name, score, timestamp))
                    with open(csv_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([pose_name, score, timestamp])
                    last_log_time = time.time()

                # Show hold timer
                cv2.putText(display,
                            f"Holding: {elapsed:.1f}s / {HOLD_TIME}s",
                            (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            (255, 255, 255),
                            3)
                if elapsed >= HOLD_TIME:
                    break  # next pose

            # Show score
            cv2.putText(display,
                        f"{pose_name}  {int(score)}%",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.4,
                        (0, 255, 0),
                        3)

            # Show textual feedback
            if mode == "coach":
                for i, line in enumerate(feedback[:4]):
                    cv2.putText(display,
                                line,
                                (20, 140 + i * 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 200, 255),
                                2)

            cv2.imshow("Yoga Sense", display)
            if cv2.waitKey(1) & 0xFF == 27:
                cap_cam.release()
                if cap_vid:
                    cap_vid.release()
                cv2.destroyAllWindows()
                return

        if cap_vid:
            cap_vid.release()

    cap_cam.release()
    cv2.destroyAllWindows()
    print(f"Session finished! Scores saved to {csv_file}")

