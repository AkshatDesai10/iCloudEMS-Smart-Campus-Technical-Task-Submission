# iCloudEMS Smart Campus - Live CV Pipeline
# Run:
#   pip install ultralytics opencv-python
#   python campus_pipeline.py
#   python campus_pipeline.py video.mp4

import cv2
import time
import sys
import numpy as np
from collections import deque
from ultralytics import YOLO


# CONFIG


SAMPLE_EVERY = 2

LOST_FRAMES = 75

# ------------------------------------------------------------
# POSTURE DETECTION - LIVE / MOTION BASED
# ------------------------------------------------------------
# No fixed aspect-ratio threshold is used.
# Posture is estimated from temporal movement of each tracked ID.

POSTURE_HISTORY_SIZE = 6
MOVEMENT_THRESHOLD = 2.5
MOVING_CONFIRM_FRAMES = 2
STATIONARY_CONFIRM_FRAMES = 4



# ------------------------------------------------------------
# OTHER SETTINGS
# ------------------------------------------------------------

BLUR_THRESHOLD = 80.0

MOTION_PIXEL_THRESHOLD = 15
MOTION_AREA_FRACTION = 0.005

BRIGHTNESS_LIGHTS_ON = 90

CONFIRM_FRAMES = 5



# SOURCE


source = sys.argv[1] if len(sys.argv) > 1 else 0

cap = cv2.VideoCapture(source)

model = YOLO("yolov8n.pt")



# VIDEO TIMING


is_file_source = source != 0

source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

frame_duration = 1.0 / source_fps




prev_gray = None

frame_idx = 0



# TRACK BOOKKEEPING


active_tracks = {}

ever_seen = set()

track_state = {}

last_event_text = "Waiting for event..."

seen_count = {}



# POSTURE BOOKKEEPING


# track_id -> recent center positions
position_history = {}

# track_id -> current posture
posture_state = {}

# track_id -> consecutive frames showing movement
moving_votes = {}

# track_id -> consecutive frames showing stability
stationary_votes = {}



# FPS


fps_smoother = deque(maxlen=30)



# MAIN LOOP


while True:

    loop_start = time.time()

    ret, frame = cap.read()

    if not ret:
        break

    frame_idx += 1


    # --------------------------------------------------------
    # Process every Nth frame
    # --------------------------------------------------------

    if frame_idx % SAMPLE_EVERY != 0:
        continue


    t0 = time.time()

    h_frame, w_frame = frame.shape[:2]


    # ========================================================
    # MOTION DETECTION
    # ========================================================

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray_small = cv2.resize(gray, (320, 180))

    gray_small = cv2.GaussianBlur(
        gray_small,
        (5, 5),
        0
    )

    motion_detected = False

    if prev_gray is not None:

        diff = cv2.absdiff(
            prev_gray,
            gray_small
        )

        _, thresh = cv2.threshold(
            diff,
            MOTION_PIXEL_THRESHOLD,
            255,
            cv2.THRESH_BINARY
        )

        changed_fraction = (
            cv2.countNonZero(thresh)
            / thresh.size
        )

        motion_detected = (
            changed_fraction > MOTION_AREA_FRACTION
        )

    prev_gray = gray_small


    # ========================================================
    # BLUR DETECTION
    # ========================================================

    lap_var = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    is_blurry = lap_var < BLUR_THRESHOLD


    # ========================================================
    # BRIGHTNESS
    # ========================================================

    mean_brightness = gray.mean()


    # ========================================================
    # YOLO + BYTETRACK
    # ========================================================

    results = model.track(
        frame,
        persist=True,
        classes=[0],
        verbose=False,
        tracker="custom_tracker.yaml",
        conf=0.25,
        iou=0.5
    )


    # IDs detected in current frame

    raw_ids_this_frame = set()


    # ========================================================
    # PERSON PROCESSING
    # ========================================================

    if results[0].boxes.id is not None:

        boxes = (
            results[0]
            .boxes
            .xyxy
            .cpu()
            .numpy()
        )

        ids = (
            results[0]
            .boxes
            .id
            .cpu()
            .numpy()
            .astype(int)
        )


        for box, tid in zip(boxes, ids):

            x1, y1, x2, y2 = box


            # ------------------------------------------------
            # Bounding box dimensions
            # ------------------------------------------------

            bw = max(x2 - x1, 1)

            bh = max(y2 - y1, 1)


            raw_ids_this_frame.add(tid)


            # =================================================
            # LIVE MOTION-BASED POSTURE
            # =================================================

            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0

            if tid not in position_history:

                position_history[tid] = deque(
                    maxlen=POSTURE_HISTORY_SIZE
                )

                posture_state[tid] = "Seated"
                moving_votes[tid] = 0
                stationary_votes[tid] = 0

            position_history[tid].append(
                (center_x, center_y)
            )

            movement = 0.0

            if len(position_history[tid]) >= 2:

                previous_x, previous_y = (
                    position_history[tid][-2]
                )

                movement = (
                    abs(center_x - previous_x)
                    + abs(center_y - previous_y)
                )


            # -------------------------------------------------
            # Temporal movement decision
            # -------------------------------------------------

            if movement >= MOVEMENT_THRESHOLD:

                moving_votes[tid] += 1
                stationary_votes[tid] = 0

                if (
                    moving_votes[tid]
                    >= MOVING_CONFIRM_FRAMES
                ):

                    posture_state[tid] = (
                        "Standing/Moving"
                    )

            else:

                stationary_votes[tid] += 1
                moving_votes[tid] = 0

                if (
                    stationary_votes[tid]
                    >= STATIONARY_CONFIRM_FRAMES
                ):

                    posture_state[tid] = "Seated"


            posture = posture_state[tid]


            # =================================================
            # ENTRY CONFIRMATION
            # =================================================

            seen_count[tid] = (
                seen_count.get(tid, 0) + 1
            )


            if (
                tid not in ever_seen
                and seen_count[tid] >= CONFIRM_FRAMES
            ):

                ever_seen.add(tid)

                track_state[tid] = "in"

                last_event_text = (
                    f"Entry detected — ID {tid}"
                )


            # =================================================
            # CONFIRMED PERSON
            # =================================================

            if tid in ever_seen:

                active_tracks[tid] = frame_idx


                # ---------------------------------------------
                # Bounding box
                # ---------------------------------------------

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )


                # ---------------------------------------------
                # Label
                # ---------------------------------------------

                label = (
                    f"ID {tid} | {posture}"
                )


                cv2.putText(
                    frame,
                    label,
                    (int(x1), int(y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )


    # ========================================================
    # EXIT DETECTION
    # ========================================================

    for tid in list(active_tracks.keys()):

        if (
            tid not in raw_ids_this_frame
            and
            frame_idx - active_tracks[tid]
            > LOST_FRAMES
        ):

            last_event_text = (
                f"Exit detected — ID {tid}"
            )

            del active_tracks[tid]

            track_state.pop(
                tid,
                None
            )

            position_history.pop(
                tid,
                None
            )

            posture_state.pop(
                tid,
                None
            )

            moving_votes.pop(
                tid,
                None
            )

            stationary_votes.pop(
                tid,
                None
            )

            seen_count.pop(
                tid,
                None
            )


    # ========================================================
    # COUNT PRESENT / STANDING / SEATED
    # ========================================================

    currently_present = len(
        active_tracks
    )

    standing_count = 0

    seated_count = 0


    for tid in active_tracks:

        posture = posture_state.get(
            tid,
            "Seated"
        )


        if posture == "Standing/Moving":

            standing_count += 1

        else:

            seated_count += 1


    room_occupied = (
        currently_present > 0
    )


    # ========================================================
    # LIGHTS ON + EMPTY ROOM
    # ========================================================

    lights_on_empty = (
        not room_occupied
        and
        mean_brightness > BRIGHTNESS_LIGHTS_ON
    )


    # ========================================================
    # FPS
    # ========================================================

    fps_smoother.append(
        1.0 /
        max(
            time.time() - t0,
            1e-6
        )
    )


    fps = (
        sum(fps_smoother)
        /
        len(fps_smoother)
    )


    # ========================================================
    # PROFESSIONAL LIVE DASHBOARD UI
    # ========================================================

    panel_width = 360
    panel_height = h_frame

    dashboard = np.zeros(
        (panel_height, panel_width, 3),
        dtype=np.uint8
    )

    # Background
    dashboard[:] = (24, 24, 28)

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    cv2.rectangle(
        dashboard,
        (0, 0),
        (panel_width, 82),
        (45, 30, 75),
        -1
    )

    cv2.putText(
        dashboard,
        "iCloudEMS",
        (22, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2
    )

    cv2.putText(
        dashboard,
        "SMART CLASSROOM",
        (22, 61),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (205, 205, 210),
        1
    )

    cv2.circle(
        dashboard,
        (316, 27),
        7,
        (0, 0, 255),
        -1
    )

    cv2.putText(
        dashboard,
        "LIVE",
        (329, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1
    )

    # --------------------------------------------------------
    # Helper functions
    # --------------------------------------------------------

    def draw_card(panel, y, title, value, subtitle=""):
        cv2.rectangle(
            panel,
            (15, y),
            (345, y + 68),
            (42, 42, 48),
            -1
        )

        cv2.putText(
            panel,
            title,
            (28, y + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (165, 165, 172),
            1
        )

        cv2.putText(
            panel,
            str(value),
            (28, y + 51),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2
        )

        if subtitle:
            cv2.putText(
                panel,
                subtitle,
                (210, y + 49),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (190, 190, 195),
                1
            )


    def draw_status_row(panel, y, title, value):
        cv2.putText(
            panel,
            title,
            (25, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (155, 155, 162),
            1
        )

        cv2.putText(
            panel,
            value,
            (195, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (240, 240, 245),
            1
        )


    # --------------------------------------------------------
    # LIVE ANALYTICS
    # --------------------------------------------------------

    cv2.putText(
        dashboard,
        "LIVE ANALYTICS",
        (20, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (195, 195, 200),
        1
    )

    draw_card(
        dashboard,
        125,
        "CURRENTLY PRESENT",
        currently_present,
        "people"
    )

    draw_card(
        dashboard,
        200,
        "SEATED",
        seated_count,
        "students"
    )

    draw_card(
        dashboard,
        275,
        "STANDING / MOVING",
        standing_count,
        "students"
    )

    draw_card(
        dashboard,
        350,
        "UNIQUE ENTRIES",
        len(ever_seen),
        "total"
    )

    # --------------------------------------------------------
    # ATTENDANCE SUMMARY
    # --------------------------------------------------------

    cv2.putText(
        dashboard,
        "ATTENDANCE",
        (20, 445),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (195, 195, 200),
        1
    )

    attendance_text = (
        f"Seated: {seated_count}   "
        f"Standing/Moving: {standing_count}"
    )

    cv2.rectangle(
        dashboard,
        (15, 458),
        (345, 492),
        (38, 38, 43),
        -1
    )

    cv2.putText(
        dashboard,
        attendance_text,
        (25, 481),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (235, 235, 240),
        1
    )

    # --------------------------------------------------------
    # SYSTEM STATUS
    # --------------------------------------------------------

    cv2.putText(
        dashboard,
        "SYSTEM STATUS",
        (20, 520),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (195, 195, 200),
        1
    )

    motion_status = (
        "Motion detected"
        if motion_detected
        else "No motion"
    )

    quality_status = (
        "Blurry - flag for review"
        if is_blurry
        else "Clear"
    )

    room_status = (
        "Occupied"
        if room_occupied
        else "Empty"
    )

    draw_status_row(
        dashboard,
        548,
        "Motion in frame",
        motion_status
    )

    draw_status_row(
        dashboard,
        576,
        "Frame quality",
        quality_status
    )

    draw_status_row(
        dashboard,
        604,
        "Room status",
        room_status
    )

    draw_status_row(
        dashboard,
        632,
        "Attendance",
        "Running"
    )

    # --------------------------------------------------------
    # LATEST EVENT
    # --------------------------------------------------------

    cv2.putText(
        dashboard,
        "LATEST EVENT",
        (20, 670),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (195, 195, 200),
        1
    )

    event_text = last_event_text

    if len(event_text) > 39:
        event_text = event_text[:39] + "..."

    cv2.rectangle(
        dashboard,
        (15, 682),
        (345, 725),
        (38, 38, 43),
        -1
    )

    cv2.putText(
        dashboard,
        event_text,
        (25, 709),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (235, 235, 240),
        1
    )

    # --------------------------------------------------------
    # Bottom status bar
    # --------------------------------------------------------

    bottom_y = panel_height - 42

    cv2.rectangle(
        dashboard,
        (0, bottom_y),
        (panel_width, panel_height),
        (32, 32, 36),
        -1
    )

    cv2.putText(
        dashboard,
        "Q  |  EXIT",
        (22, bottom_y + 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (175, 175, 180),
        1
    )

    # --------------------------------------------------------
    # CCTV label
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (0, 0),
        (210, 34),
        (35, 35, 35),
        -1
    )

    cv2.putText(
        frame,
        "LIVE CCTV FEED",
        (12, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1
    )

    # --------------------------------------------------------
    # SHOW CCTV + DASHBOARD IN ONE WINDOW
    # --------------------------------------------------------

    # Keep the complete CCTV feed visible on the left and place
    # the dashboard beside it in the same application window.
    combined_width = w_frame + panel_width

    combined = np.zeros(
        (h_frame, combined_width, 3),
        dtype=np.uint8
    )

    # Full CCTV video on the left.
    combined[:, :w_frame] = frame

    # Dashboard on the right.
    combined[:, w_frame:] = dashboard

    cv2.line(
        combined,
        (w_frame, 0),
        (w_frame, h_frame),
        (90, 90, 95),
        2
    )

    cv2.imshow(
        "iCloudEMS Smart Classroom",
        combined
    )

    # ========================================================
    # REAL-TIME VIDEO THROTTLING
    # ========================================================

    if is_file_source:

        target_elapsed = (
            frame_duration
            * SAMPLE_EVERY
        )

        actual_elapsed = (
            time.time()
            - loop_start
        )

        remaining = (
            target_elapsed
            - actual_elapsed
        )


        if remaining > 0:

            time.sleep(
                remaining
            )


    # ========================================================
    # QUIT
    # ========================================================

    if (
        cv2.waitKey(1) & 0xFF
        == ord("q")
    ):

        break



# CLEANUP


cap.release()

cv2.destroyAllWindows()