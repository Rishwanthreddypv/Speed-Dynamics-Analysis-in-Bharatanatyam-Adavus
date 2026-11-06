import pipeline_paths as paths
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rtmlib import Wholebody
FPS = 30

DEVICE = "cpu"
BACKEND = "onnxruntime"

CONFIDENCE_THRESHOLD = 0.30
FOOT_KEYPOINTS = {

        17: "Left_Big_Toe",

        18: "Left_Small_Toe",

        19: "Left_Heel",

        20: "Right_Big_Toe",

        21: "Right_Small_Toe",

        22: "Right_Heel"

    }


    # ============================================================
    # COLORS
    # ============================================================

COLORS = {

        17: (0, 255, 0),

        18: (0, 200, 0),

        19: (255, 0, 0),

        20: (0, 0, 255),

        21: (0, 0, 200),

        22: (0, 255, 255)

    }




def run():
    FRAMES_DIR = paths.FRAMES_DIR

# ============================================================
# INPUT
# ============================================================



# ============================================================
# OUTPUT
# ============================================================
    OUTPUT_DIR = paths.RTM_DIR
    if not os.path.exists(FRAMES_DIR):
        raise FileNotFoundError(
            f"Frames directory not found:\n{FRAMES_DIR}"
    )
    GRAPH_DIR = os.path.join(
        OUTPUT_DIR,
        "graphs"
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    os.makedirs(
        GRAPH_DIR,
        exist_ok=True
    )


    EXCEL_OUT = os.path.join(
        OUTPUT_DIR,
        "foot_keypoints.xlsx"
    )

    VIDEO_OUT = os.path.join(
        OUTPUT_DIR,
        "foot_tracking.mp4"
    )


    # ============================================================
    # SETTINGS
    # ============================================================


    # ============================================================
    # COCO WHOLEBODY FOOT KEYPOINTS
    # ============================================================

    
    # ============================================================
    # INITIALIZE WHOLEBODY MODEL
    # ============================================================

    print("\n==========================================")

    print("Initializing RTMPose WholeBody...")

    print("==========================================\n")


    wholebody = Wholebody(

        to_openpose=False,

        mode="balanced",

        backend=BACKEND,

        device=DEVICE

    )


    print("\nRTMPose WholeBody initialized.\n")


    # ============================================================
    # GET FRAME FILES
    # ============================================================

    frame_files = sorted(

        [

            file

            for file in os.listdir(FRAMES_DIR)

            if file.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png"
                )
            )

        ]

    )


    if len(frame_files) == 0:

        raise RuntimeError(
            "No frames found."
        )


    print(
        f"Frames Found : {len(frame_files)}"
    )


    # ============================================================
    # READ FIRST FRAME
    # ============================================================

    first_frame_path = os.path.join(

        FRAMES_DIR,

        frame_files[0]

    )


    first_frame = cv2.imread(
        first_frame_path
    )


    if first_frame is None:

        raise RuntimeError(
            "Unable to read first frame."
        )


    height, width = first_frame.shape[:2]


    print(
        f"Resolution : {width} x {height}"
    )


    # ============================================================
    # VIDEO WRITER
    # ============================================================

    writer = cv2.VideoWriter(

        VIDEO_OUT,

        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),

        FPS,

        (
            width,
            height
        )

    )


    if not writer.isOpened():

        raise RuntimeError(
            "Unable to initialize video writer."
        )


    # ============================================================
    # DATA STORAGE
    # ============================================================

    rows = []


    # ============================================================
    # PRINT SHAPE ONLY ONCE
    # ============================================================

    shape_printed = False


    # ============================================================
    # PROCESS FRAMES
    # ============================================================

    for frame_id, frame_file in enumerate(
        frame_files
    ):

        frame_path = os.path.join(

            FRAMES_DIR,

            frame_file

        )


        frame = cv2.imread(
            frame_path
        )


        if frame is None:

            print(
                f"Cannot read frame: {frame_file}"
            )

            continue


        # ========================================================
        # WHOLEBODY INFERENCE
        # ========================================================

        keypoints, scores = wholebody(
            frame
        )


        # ========================================================
        # DEBUG SHAPE
        # ========================================================

        if not shape_printed:

            print(
                "\nKeypoints shape:",
                np.shape(keypoints)
            )

            print(
                "Scores shape:",
                np.shape(scores)
            )

            shape_printed = True


        # ========================================================
        # INITIALIZE ROW
        # ========================================================

        row = {

            "Frame": frame_id,

            "Time(s)": frame_id / FPS

        }


        # ========================================================
        # INITIALIZE ALL FOOT VALUES AS NAN
        # ========================================================

        for keypoint_name in FOOT_KEYPOINTS.values():

            row[
                f"{keypoint_name}_X"
            ] = np.nan


            row[
                f"{keypoint_name}_Y"
            ] = np.nan


            row[
                f"{keypoint_name}_Confidence"
            ] = np.nan


        # ========================================================
        # CHECK PERSON DETECTION
        # ========================================================

        if (

            keypoints is not None

            and scores is not None

            and len(keypoints) > 0

        ):

            # ====================================================
            # FIRST DETECTED PERSON
            # ====================================================

            person_keypoints = keypoints[0]

            person_scores = scores[0]


            # ====================================================
            # SAFETY CHECK
            # ====================================================

            number_of_keypoints = len(
                person_keypoints
            )


            if number_of_keypoints < 23:

                print(

                    f"Frame {frame_id}: "
                    f"Only {number_of_keypoints} "
                    f"keypoints returned."

                )


            else:

                # ================================================
                # PROCESS FOOT KEYPOINTS
                # ================================================

                for (
                    keypoint_id,
                    keypoint_name
                ) in FOOT_KEYPOINTS.items():


                    # ============================================
                    # GET COORDINATES
                    # ============================================

                    x = float(

                        person_keypoints[
                            keypoint_id
                        ][0]

                    )


                    y = float(

                        person_keypoints[
                            keypoint_id
                        ][1]

                    )


                    confidence = float(

                        person_scores[
                            keypoint_id
                        ]

                    )


                    # ============================================
                    # STORE VALUES
                    # ============================================

                    row[
                        f"{keypoint_name}_X"
                    ] = x


                    row[
                        f"{keypoint_name}_Y"
                    ] = y


                    row[
                        f"{keypoint_name}_Confidence"
                    ] = confidence


                    # ============================================
                    # DRAW CONFIDENT KEYPOINT
                    # ============================================

                    if confidence >= CONFIDENCE_THRESHOLD:

                        color = COLORS[
                            keypoint_id
                        ]


                        # ========================================
                        # KEYPOINT
                        # ========================================

                        cv2.circle(

                            frame,

                            (
                                int(x),
                                int(y)
                            ),

                            10,

                            color,

                            -1

                        )


                        # ========================================
                        # WHITE BORDER
                        # ========================================

                        cv2.circle(

                            frame,

                            (
                                int(x),
                                int(y)
                            ),

                            15,

                            (
                                255,
                                255,
                                255
                            ),

                            2

                        )


                        # ========================================
                        # LABEL
                        # ========================================

                        cv2.putText(

                            frame,

                            (
                                f"{keypoint_id}: "
                                f"{keypoint_name}"
                            ),

                            (
                                int(x) + 15,
                                int(y)
                            ),

                            cv2.FONT_HERSHEY_SIMPLEX,

                            0.55,

                            color,

                            2

                        )


                        # ========================================
                        # CONFIDENCE
                        # ========================================

                        cv2.putText(

                            frame,

                            f"Conf: {confidence:.2f}",

                            (
                                int(x) + 15,
                                int(y) + 25
                            ),

                            cv2.FONT_HERSHEY_SIMPLEX,

                            0.50,

                            color,

                            2

                        )


        # ========================================================
        # NO PERSON DETECTED
        # ========================================================

        else:

            print(

                f"Frame {frame_id}: "
                f"No person detected"

            )


        # ========================================================
        # STORE ROW
        # ========================================================

        rows.append(
            row
        )


        # ========================================================
        # FRAME INFORMATION
        # ========================================================

        cv2.putText(

            frame,

            f"Frame: {frame_id}",

            (
                20,
                40
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (
                255,
                255,
                255
            ),

            2

        )


        cv2.putText(

            frame,

            "RTMPose WholeBody - Foot Keypoints",

            (
                20,
                80
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (
                255,
                255,
                255
            ),

            2

        )


        cv2.putText(

            frame,

            "17-19: LEFT FOOT | 20-22: RIGHT FOOT",

            (
                20,
                115
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (
                255,
                255,
                255
            ),

            2

        )


        # ========================================================
        # WRITE VIDEO
        # ========================================================

        writer.write(
            frame
        )


        # ========================================================
        # DISPLAY
        # ========================================================

        display_width = 1280

        scale = display_width / width

        display_height = int(
            height * scale
        )


        display = cv2.resize(

            frame,

            (
                display_width,
                display_height
            )

        )


        cv2.imshow(

            "RTMPose WholeBody Foot Tracking",

            display

        )


        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):

            print(
                "\nTracking stopped by user."
            )

            break


        # ========================================================
        # PROGRESS
        # ========================================================

        print(

            f"Processed Frame "
            f"{frame_id + 1}/"
            f"{len(frame_files)}"

        )


    # ============================================================
    # RELEASE
    # ============================================================

    writer.release()

    cv2.destroyAllWindows()


    # ============================================================
    # DATAFRAME
    # ============================================================

    df = pd.DataFrame(
        rows
    )


    # ============================================================
    # SAVE EXCEL
    # ============================================================

    df.to_excel(

        EXCEL_OUT,

        index=False

    )


    # ============================================================
    # INDIVIDUAL Y COORDINATE GRAPHS
    # ============================================================

    for (
        keypoint_id,
        keypoint_name
    ) in FOOT_KEYPOINTS.items():


        y_column = (
            f"{keypoint_name}_Y"
        )


        if y_column not in df.columns:

            continue


        plt.figure(

            figsize=(
                16,
                6
            )

        )


        plt.plot(

            df["Frame"],

            df[y_column],

            linewidth=2

        )


        plt.xlabel(
            "Frame Number"
        )


        plt.ylabel(
            "Y Coordinate (Pixels)"
        )


        plt.title(

            f"Frame vs {keypoint_name} Y Coordinate"

        )


        plt.grid(
            True
        )


        plt.tight_layout()


        graph_path = os.path.join(

            GRAPH_DIR,

            f"frame_vs_{keypoint_name}_y.png"

        )


        plt.savefig(

            graph_path,

            dpi=600

        )


        plt.close()


    # ============================================================
    # COMBINED GRAPH
    # ============================================================

    plt.figure(

        figsize=(
            18,
            7
        )

    )


    for (
        keypoint_id,
        keypoint_name
    ) in FOOT_KEYPOINTS.items():


        y_column = (
            f"{keypoint_name}_Y"
        )


        if y_column in df.columns:

            plt.plot(

                df["Frame"],

                df[y_column],

                linewidth=1.5,

                label=keypoint_name

            )


    plt.xlabel(
        "Frame Number"
    )


    plt.ylabel(
        "Y Coordinate (Pixels)"
    )


    plt.title(

        "RTMPose WholeBody Foot Keypoint Y Coordinates"

    )


    plt.grid(
        True
    )


    plt.legend()


    plt.tight_layout()


    COMBINED_GRAPH = os.path.join(

        GRAPH_DIR,

        "combined_foot_y_coordinates.png"

    )


    plt.savefig(

        COMBINED_GRAPH,

        dpi=600

    )


    plt.close()


    # ============================================================
    # CONFIDENCE SUMMARY
    # ============================================================

    print("\n==========================================")

    print("FOOT KEYPOINT CONFIDENCE SUMMARY")

    print("==========================================")


    for keypoint_name in FOOT_KEYPOINTS.values():

        confidence_column = (

            f"{keypoint_name}_Confidence"

        )


        mean_confidence = df[

            confidence_column

        ].mean()


        print(

            f"{keypoint_name:<20} : "
            f"{mean_confidence:.4f}"

        )


    # ============================================================
    # OUTPUT
    # ============================================================

    print("\n==========================================")

    print("RTMPOSE WHOLEBODY TRACKING COMPLETED")

    print("==========================================")

    print(
        f"Excel : {EXCEL_OUT}"
    )

    print(
        f"Video : {VIDEO_OUT}"
    )

    print(
        f"Graphs: {GRAPH_DIR}"
    )

    print("==========================================")
    return OUTPUT_DIR