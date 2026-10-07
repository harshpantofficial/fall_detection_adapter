import spaces
import cv2
import numpy as np
import gradio as gr
import os
import sys
import tempfile
from collections import deque

if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fall_detection_core import (
    MODEL_PATH,
    YOLO_POSE_MODEL,
    INPUT_TIMESTEPS,
    FALL_CONFIDENCE_THRESHOLD,
    MIN_KEYPOINT_CONFIDENCE_FOR_NORMALIZATION,
    NUM_KEYPOINTS_TRAINING,
    NUM_FEATURES,
    SORTED_YOUR_KEYPOINT_NAMES,
    KEYPOINT_DICT_TRAINING,
    YOLO_IDX_TO_NAME,
    normalize_skeleton_frame,
    extract_features_from_yolo_result,
    load_models,
)

FALL_EVENT_COOLDOWN = 10  # seconds — used as (cooldown * fps) frames in video loop

print("--- Initializing ---")
print(f"NUM_FEATURES: {NUM_FEATURES}")

# --- Load models at startup (reuse shared loader) ---
yolo_pose, interpreter, input_details, output_details = load_models()
model_expected_shape = tuple(input_details[0]["shape"])
print(f"Model expected input shape: {model_expected_shape}")


@spaces.GPU(duration=300)  # allocate GPU for the full video processing window
def run_fall_detection_on_video(video_path, progress=gr.Progress(track_tqdm=True)):
    if video_path is None:
        return None, "Please upload a video file."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, f"Error: Cannot open video: {video_path}"

    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Unique temp file per request — prevents concurrent users overwriting each other
    fd, output_path = tempfile.mkstemp(suffix=".mp4", prefix="fall_detection_", dir=tempfile.gettempdir())
    os.close(fd)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H.264 — playable in browsers
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    # Fallback: if avc1 is unavailable (OpenCV build without H.264), try mp4v
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_buffer             = deque(maxlen=INPUT_TIMESTEPS)
    fall_event_count         = 0
    last_fall_frame          = -FALL_EVENT_COOLDOWN * fps
    frame_idx                = 0
    fall_detected_this_frame = False
    fall_confidence_val      = 0.0

    print(f"Processing video: {video_path} ({total_frames} frames, {fps:.1f} fps)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        fall_detected_this_frame = False

        results = yolo_pose(frame, verbose=False, device="cuda")
        result  = results[0]

        features = extract_features_from_yolo_result(result, frame_width, frame_height)
        frame_buffer.append(features)

        if len(frame_buffer) == INPUT_TIMESTEPS:
            input_seq = np.array(frame_buffer, dtype=np.float32)[np.newaxis, ...]
            interpreter.set_tensor(input_details[0]["index"], input_seq)
            interpreter.invoke()
            output_data         = interpreter.get_tensor(output_details[0]["index"])
            fall_confidence_val = float(output_data[0][0])

            if (fall_confidence_val >= FALL_CONFIDENCE_THRESHOLD and
                    (frame_idx - last_fall_frame) >= FALL_EVENT_COOLDOWN * fps):
                fall_detected_this_frame = True
                fall_event_count        += 1
                last_fall_frame          = frame_idx
                print(f"  FALL DETECTED at frame {frame_idx} | confidence: {fall_confidence_val:.4f}")

        annotated_frame = result.plot(kpt_radius=4, line_width=2)

        status_text   = "FALL DETECTED!" if fall_detected_this_frame else "Normal"
        conf_text     = f"Conf: {fall_confidence_val:.2f}" if len(frame_buffer) == INPUT_TIMESTEPS else "Warming up..."
        overlay_color = (0, 0, 255) if fall_detected_this_frame else (0, 200, 0)

        cv2.rectangle(annotated_frame, (0, 0), (350, 70), (0, 0, 0), -1)
        cv2.putText(annotated_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, overlay_color, 2)
        cv2.putText(annotated_frame, conf_text, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        cv2.rectangle(annotated_frame, (frame_width - 200, 0), (frame_width, 40), (0, 0, 0), -1)
        cv2.putText(annotated_frame, f"Falls: {fall_event_count}", (frame_width - 190, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)

        out.write(annotated_frame)

    cap.release()
    out.release()

    summary = (
        f"Processing complete!\n"
        f"Total frames processed : {frame_idx}\n"
        f"Fall events detected   : {fall_event_count}\n"
        f"Threshold used         : {FALL_CONFIDENCE_THRESHOLD:.0%}\n"
        f"Model                  : YOLO26-pose + TFLite Transformer"
    )
    print(summary)
    return output_path, summary


EXAMPLE_VIDEOS = [f for f in [
    "fall_example_1.mp4", "fall_example_2.mp4",
    "fall_example_3.mp4", "fall_example_4.mp4"
] if os.path.exists(f)]

with gr.Blocks(title="Fall Detection - YOLO26 + Transformer") as demo:

    gr.Markdown(
        """
        # Fall Detection System
        ### YOLO26-pose keypoint extraction + TFLite Transformer classifier
        Upload a video. The system extracts 17-keypoint poses with YOLO26,
        feeds a 30-frame sliding window into the Transformer, and flags falls >= 90% confidence.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload Video", sources=["upload"])
            run_btn = gr.Button("Detect Falls", variant="primary", size="lg")
            gr.Markdown(
                f"**Pose Extractor:** YOLO26n-pose  \n"
                f"**Classifier:** TFLite Transformer  \n"
                f"**Sequence:** {INPUT_TIMESTEPS} frames  \n"
                f"**Threshold:** {FALL_CONFIDENCE_THRESHOLD:.0%}  \n"
                f"**Keypoints:** {NUM_KEYPOINTS_TRAINING} (COCO-17)"
            )
        with gr.Column(scale=1):
            video_output = gr.Video(label="Annotated Output")
            summary_output = gr.Textbox(label="Detection Summary", lines=6, interactive=False)

    if EXAMPLE_VIDEOS:
        gr.Markdown("### Example Videos")
        gr.Examples(examples=[[v] for v in EXAMPLE_VIDEOS], inputs=video_input, label="Click to load")

    run_btn.click(
        fn=run_fall_detection_on_video,
        inputs=[video_input],
        outputs=[video_output, summary_output],
        show_progress=True
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        ssr_mode=False,
        theme=gr.themes.Soft(primary_hue="red"),
        css="footer { display: none !important; }",
    )