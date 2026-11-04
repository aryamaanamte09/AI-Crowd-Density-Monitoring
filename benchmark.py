
import time
import torch
import cv2
from ultralytics import YOLO

VIDEO_PATH = "videos/People Walking Free Stock Footage, Royalty-Free No Copyright Content.mp4"  # Change this to your test video path
MODEL_PATH = "yolo26n.pt"
IMAGE_SIZE = 640
FRAME_STRIDE = 3


def benchmark(device):
    print(f"\nRunning benchmark on: {device.upper()}")

    model = YOLO(MODEL_PATH)

    # Warm up the model before measuring
    dummy = torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE))
    if device == "cuda":
        torch.cuda.synchronize()

    start_warmup = time.perf_counter()
    model.predict(
        source="https://ultralytics.com/images/bus.jpg",
        device=device,
        imgsz=IMAGE_SIZE,
        verbose=False
    )
    if device == "cuda":
        torch.cuda.synchronize()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

    frame_count = 0
    total_inference_ms = 0.0
    start = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Match the project's frame-stride setting
            if frame_count % FRAME_STRIDE != 0:
                frame_count += 1
                continue

            if device == "cuda":
                torch.cuda.synchronize()

            inference_start = time.perf_counter()
            model.predict(
                source=frame,
                device=device,
                imgsz=IMAGE_SIZE,
                verbose=False
            )

            if device == "cuda":
                torch.cuda.synchronize()

            total_inference_ms += (
                time.perf_counter() - inference_start
            ) * 1000

            frame_count += 1

    finally:
        cap.release()

    elapsed = time.perf_counter() - start
    processed_frames = (frame_count + FRAME_STRIDE - 1) // FRAME_STRIDE

    # Count processed frames accurately when the video has frames
    processed_frames = max(processed_frames, 0)

    if processed_frames == 0:
        print("No frames were processed. Check the video path.")
        return None

    avg_ms = total_inference_ms / processed_frames
    fps = processed_frames / elapsed

    print(f"Device: {device.upper()}")
    print(f"Processed frames: {processed_frames}")
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Average inference time: {avg_ms:.2f} ms/frame")
    print(f"Throughput: {fps:.2f} processed frames/second")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "device": device.upper(),
        "frames": processed_frames,
        "total_time": elapsed,
        "avg_ms": avg_ms,
        "fps": fps
    }


if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())

    results = []
    results.append(benchmark("cpu"))

    if torch.cuda.is_available():
        results.append(benchmark("cuda"))
    else:
        print("CUDA is unavailable. GPU benchmark skipped.")

    if len(results) == 2 and results[0] and results[1]:
        cpu, gpu = results
        print("\n========== COMPARISON ==========")
        print(f"CPU time: {cpu['total_time']:.2f} s")
        print(f"GPU time: {gpu['total_time']:.2f} s")
        print(f"CPU average inference: {cpu['avg_ms']:.2f} ms/frame")
        print(f"GPU average inference: {gpu['avg_ms']:.2f} ms/frame")
        print(f"CPU throughput: {cpu['fps']:.2f} FPS")
        print(f"GPU throughput: {gpu['fps']:.2f} FPS")

        if gpu["total_time"] > 0:
            print(
                f"Overall speedup: "
                f"{cpu['total_time'] / gpu['total_time']:.2f}x"
            )
