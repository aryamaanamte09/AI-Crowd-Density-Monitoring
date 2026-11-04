
import os
import cv2
import av
import uuid
import math
import hashlib
import tempfile
import threading
import streamlit as st
import subprocess
import shutil

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from streamlit_webrtc import webrtc_streamer
from detector import analyse_video, OBJECT_CLASSES, model

def make_camera_callback(selected_objects, camera_state, camera_lock):
    #convert selected display names into YOLO class names
    selected_names = [OBJECT_CLASSES[name] for name in selected_objects]

    class_ids = [
        class_id
        for class_id, class_name in model.names.items()
        if class_name in selected_names
    ]

    def video_frame_callback(frame):
        image = frame.to_ndarray(format="bgr24")

        try:
            #run detection on every incoming camera frame
            result = model.track(
                source=image,
                persist=True,
                tracker="bytetrack_custom.yaml",
                classes=class_ids,
                conf=0.25,
                imgsz=320,
                device=DEVICE,
                verbose=False
            )[0]

            counts = {name: 0 for name in selected_objects}

            #count objects by their selected class
            for box in result.boxes:
                class_id = int(box.cls[0])
                detected_name = model.names[class_id]

                for display_name, actual_name in OBJECT_CLASSES.items():
                    if (
                        display_name in counts
                        and actual_name == detected_name
                    ):
                        counts[display_name] += 1

            total_count = sum(counts.values())

            #calculate density for the current frame
            if total_count <= 5:
                density = "LOW"
            elif total_count <= 15:
                density = "MEDIUM"
            elif total_count <= 30:
                density = "HIGH"
            else:
                density = "CRITICAL"

            #draw detection boxes and labels
            annotated_frame = result.plot(
                line_width=1,
                labels=True,
                conf=False
            )

            #draw current count and density over the live feed
            cv2.rectangle(
                annotated_frame,
                (10, 10),
                (310, 95),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                annotated_frame,
                f"Total Count: {total_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                annotated_frame,
                f"Density: {density}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            #share the latest results with the Streamlit interface
            with camera_lock:
                camera_state["counts"] = counts.copy()
                camera_state["total"] = total_count
                camera_state["density"] = density
                camera_state["error"] = None

            return av.VideoFrame.from_ndarray(
                annotated_frame,
                format="bgr24"
            )

        except Exception as error:
            with camera_lock:
                camera_state["error"] = str(error)

            #keep the camera feed visible if detection fails
            return av.VideoFrame.from_ndarray(
                image,
                format="bgr24"
            )

    return video_frame_callback

st.set_page_config(
    page_title="AI Crowd Monitoring",
    layout="wide"
)

st.title("AI Crowd Monitoring")

#choose the video source
source_type = st.radio(
    "Source",
    ["Video file", "Live camera"],
    horizontal=True
)

#choose the objects to detect
selected_objects = st.multiselect(
    "Detect",
    options=list(OBJECT_CLASSES.keys()),
    default=["People"]
)

#shared camera data for the live detection callback
if "camera_shared" not in st.session_state:
    st.session_state["camera_shared"] = {
        "lock": threading.Lock(),
        "selected_objects": ["People"],
        "counts": {}
    }

camera_shared = st.session_state["camera_shared"]


def get_class_ids(object_names):
    #convert selected object names into yolo class ids
    selected_names = [
        OBJECT_CLASSES[name] for name in object_names
    ]

    return [
        class_id
        for class_id, class_name in model.names.items()
        if class_name in selected_names
    ]


def process_video(input_path, object_names):
    #get the original video's frame rate
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise ValueError("Could not open this video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0

    #create a temporary file for the processed video
    output_path = os.path.join(
        tempfile.gettempdir(),
        f"crowd_output_{uuid.uuid4().hex}.mp4"
    )

    writer = None
    frame_count = 0
    total_class_counts = {name: 0 for name in object_names}
    density_history = []

    try:
        for frame, counts in analyse_video(input_path, object_names):
            if writer is None:
                height, width = frame.shape[:2]

                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    max(fps / 3, 1.0),
                    (width, height)
                )

                if not writer.isOpened():
                    raise RuntimeError("Could not create the processed video")

            writer.write(frame)
            # accumulate detections from every analysed frame
            for name, count in counts.items():
                total_class_counts[name] += count

            # record density for this frame
            total_count = sum(counts.values())

            if total_count <= 5:
                density = "LOW"
            elif total_count <= 15:
                density = "MEDIUM"
            elif total_count <= 30:
                density = "HIGH"
            else:
                density = "CRITICAL"

            density_history.append(density)
            frame_count += 1

    finally:
        if writer is not None:
            writer.release()

    if frame_count == 0:
        raise ValueError("No frames could be processed")

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("The processed video file is empty")

    st.write("Processed video size:", os.path.getsize(output_path), "bytes")
    st.write("Processed frames:", frame_count)

    return output_path, total_class_counts, density_history, frame_count


def video_frame_callback(frame):
    #read the current camera settings
    with camera_shared["lock"]:
        object_names = list(camera_shared["selected_objects"])

    image = frame.to_ndarray(format="bgr24")

    if not object_names:
        return frame

    class_ids = get_class_ids(object_names)

    #run yolo on the live camera frame
    result = model.predict(
        source=image,
        classes=class_ids,
        conf=0.25,
        verbose=False
    )[0]

    #count the selected objects in this frame
    counts = {name: 0 for name in object_names}

    for box in result.boxes:
        class_name = model.names[int(box.cls[0])]

        for display_name, actual_name in OBJECT_CLASSES.items():
            if (
                actual_name == class_name
                and display_name in counts
            ):
                counts[display_name] += 1

    #share the current counts with the dashboard
    with camera_shared["lock"]:
        camera_shared["counts"] = counts.copy()

    #return the frame with bounding boxes
    annotated_frame = result.plot(
        line_width=1,
        labels=True,
        conf=False
    )

    # calculate total detected objects
    total_count = sum(counts.values())

    # assign density level
    if total_count <= 5:
        density = "LOW"
    elif total_count <= 15:
        density = "MEDIUM"
    elif total_count <= 30:
        density = "HIGH"
    else:
        density = "CRITICAL"

    # show count and density on the video
    cv2.rectangle(annotated_frame, (10, 10), (260, 85), (0, 0, 0), -1)

    cv2.putText(
        annotated_frame,
        f"Vehicle Count: {total_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Density: {density}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    yield annotated_frame, counts

    return av.VideoFrame.from_ndarray(
        annotated_frame,
        format="bgr24"
    )


@st.fragment(run_every=1)
def show_camera_counts():
    #refresh the object counts while the camera is running
    with camera_shared["lock"]:
        counts = camera_shared["counts"].copy()

    if counts:
        columns = st.columns(len(counts))

        for column, (name, count) in zip(columns, counts.items()):
            column.metric(name, count)


if source_type == "Video file":
    uploaded_file = st.file_uploader(
        "Choose a video",
        type=[
            "mp4", "avi", "mov", "mkv",
            "mpeg", "mpg", "webm", "m4v"
        ]
    )

    if st.button(
        "Process video",
        type="primary",
        disabled=uploaded_file is None or not selected_objects
    ):
        input_path = None

        try:
            #save the upload temporarily for yolo to read
            suffix = os.path.splitext(uploaded_file.name)[1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                input_path = temp_file.name

            with st.spinner("Processing video..."):
                output_path, total_class_counts, density_history, frame_count = process_video(
                    input_path, selected_objects
                )

                ffmpeg_path = shutil.which("ffmpeg")

                if ffmpeg_path is None:
                    st.error("FFmpeg is not installed or is not on PATH.")
                else:
                    browser_video_path = output_path.replace(".mp4", "_h264.mp4")

                    conversion = subprocess.run(
                        [
                            ffmpeg_path,
                            "-y",
                            "-i", output_path,
                            "-c:v", "libx264",
                            "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart",
                            "-an",
                            browser_video_path
                        ],
                        capture_output=True,
                        text=True
                    )

                    if conversion.returncode != 0:
                        st.error("Video conversion failed.")
                        st.code(conversion.stderr[-2500:])
                    else:
                        st.subheader("Processed Video")
                        st.video(browser_video_path)

            #remove the previous output after the new one succeeds
            old_path = st.session_state.get("processed_video_path")

            st.session_state["processed_video_path"] = output_path
            st.session_state["video_counts"] = total_class_counts
            st.session_state["density_history"] = density_history
            st.session_state["frame_count"] = frame_count
            st.session_state["video_signature"] = (
                hashlib.sha256(uploaded_file.getvalue()).hexdigest(),
                tuple(sorted(selected_objects))
            )

            if old_path and os.path.exists(old_path):
                if old_path != output_path:
                    os.remove(old_path)

        except Exception as error:
            st.error(f"Could not process video: {error}")

        finally:
            if input_path and os.path.exists(input_path):
                os.remove(input_path)

                # show the processed video summary without duplicating the video player
                if uploaded_file is not None and selected_objects:
                    current_signature = (
                        hashlib.sha256(uploaded_file.getvalue()).hexdigest(),
                        tuple(sorted(selected_objects))
                    )
        # show the processed video summary without duplicating the video player

        if uploaded_file is not None and selected_objects:
            current_signature = (
                hashlib.sha256(uploaded_file.getvalue()).hexdigest(),
                tuple(sorted(selected_objects))
            )

            if st.session_state.get("video_signature") == current_signature:
                total_class_counts = st.session_state.get("video_counts", {})
                density_history = st.session_state.get("density_history", [])
                frame_count = st.session_state.get("frame_count", 0)

                # show average detections per analysed frame
                st.subheader("Average Objects Per Frame")

                if frame_count > 0:
                    for name, total in total_class_counts.items():
                        average = total / frame_count
                        st.metric(f"Average {name} per frame", math.floor(average))

                if density_history and frame_count > 0:
                    from collections import Counter

                    density_counts = Counter(density_history)

                    st.subheader("Density Summary")
                    st.metric(
                        "Most Frequent Density",
                        density_counts.most_common(1)[0][0]
                    )
                    st.metric("Frames Analysed", frame_count)

                    for level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
                        count = density_counts.get(level, 0)
                        percentage = count / frame_count * 100
                        st.write(f"{level}: {count} frames ({percentage:.1f}%)")

elif source_type == "Live camera":
    if not selected_objects:
        st.warning("Select at least one object class first.")

    else:
        # share live results between the camera callback and the UI
        if "live_camera_stats" not in st.session_state:
            st.session_state["live_camera_stats"] = {
                "counts": {},
                "total": 0,
                "density": "LOW",
                "error": None
            }

        if "live_camera_lock" not in st.session_state:
            st.session_state["live_camera_lock"] = threading.Lock()

        camera_state = st.session_state["live_camera_stats"]
        camera_lock = st.session_state["live_camera_lock"]

        # use a new stream key when the selected classes change
        classes_signature = hashlib.sha256(
            "|".join(sorted(selected_objects)).encode()
        ).hexdigest()[:8]

        callback = make_camera_callback(
            selected_objects,
            camera_state,
            camera_lock
        )

        webrtc_streamer(
            key=f"live-camera-{classes_signature}",
            video_frame_callback=callback,
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            async_processing=True
        )


        # refresh the displayed counts while the camera is running
        @st.fragment(run_every=1)
        def show_live_results():
            with camera_lock:
                current_counts = camera_state["counts"].copy()
                current_total = camera_state["total"]
                current_density = camera_state["density"]
                current_error = camera_state["error"]

            st.subheader("Live Detection Results")
            st.metric("Current Object Count", current_total)
            st.metric("Current Density", current_density)

            for name, count in current_counts.items():
                st.write(f"{name}: {count}")

            if current_error:
                st.warning(f"Detection error: {current_error}")


        show_live_results()