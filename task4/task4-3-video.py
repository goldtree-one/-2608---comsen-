from __future__ import annotations

import argparse
import time

import cv2

from task4_common import (Assets, add_asset_argument, add_predict_arguments,
                          check_file, find_asset_dir, iou, load_model,
                          print_device_info, save_json, summarize_speed)


def compute_tracking_metrics(results) -> dict:
    track_ids: dict[int, int] = {}
    per_frame_count: list[int] = []
    prev_boxes: dict[int, object] = {}
    iou_seq: list[float] = []

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            per_frame_count.append(0)
            continue
        per_frame_count.append(len(boxes))

        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
        xyxy = boxes.xyxy.cpu().numpy()

        current = {}
        for i, box in enumerate(xyxy):
            tid = int(ids[i]) if ids is not None else i
            current[tid] = box
            track_ids[tid] = track_ids.get(tid, 0) + 1
            if tid in prev_boxes:
                iou_seq.append(iou(prev_boxes[tid], box))
        prev_boxes = current

    n_frames = len(per_frame_count)
    frames_with_target = sum(1 for c in per_frame_count if c > 0)
    return {
        "frames": n_frames,
        "unique_track_ids": len(track_ids),
        "detection_rate": round(frames_with_target / n_frames, 4) if n_frames else 0.0,
        "mean_objects_per_frame": round(float(sum(per_frame_count) / n_frames), 2)
        if n_frames else 0.0,
        "max_objects_in_a_frame": int(max(per_frame_count)) if per_frame_count else 0,
        "mean_consecutive_iou": round(float(sum(iou_seq) / len(iou_seq)), 4)
        if iou_seq else None,
        "mean_track_length": round(float(sum(track_ids.values()) / len(track_ids)), 2)
        if track_ids else 0.0,
        "note": ("EAO 需要逐帧真值标注，本视频无真值，故未计算；"
                 "上列为无需真值的追踪质量代理指标。"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task4-3 无人机视频推理与追踪")
    add_asset_argument(parser)
    add_predict_arguments(parser)
    parser.add_argument("--video", default="example.mp4",
                        help="example 目录下的视频文件名，默认 example.mp4")
    parser.add_argument("--tracker", default="botsort.yaml",
                        help="追踪器配置，默认 botsort.yaml；也可用 bytetrack.yaml")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    print("=" * 70)
    print("Task4-3 无人机视频推理与追踪")
    print("=" * 70)

    assets = Assets(find_asset_dir(args.asset_dir))
    assets.describe()
    print_device_info(args.device)

    video = check_file(assets.example_dir / args.video, "请确认 example 目录下有 example.mp4")
    model = load_model(assets.drone_weights, "drone.pt 应在素材目录下")

    cap = cv2.VideoCapture(str(video))
    fps_src = cap.get(cv2.CAP_PROP_FPS)
    n_frames_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"\n[视频] {video.name}: {width}x{height}, {fps_src:.2f} FPS, "
          f"{n_frames_src} 帧, 时长 {n_frames_src / fps_src:.1f}s")

    save_dir = args.output / "predict_video"
    print("\n[推理] 使用 model.track 进行追踪 ...")

    results = model.track(
        source=str(video),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        tracker=args.tracker,
        persist=True,
        stream=True,
        save=True,
        project=str(save_dir.parent),
        name=save_dir.name,
        exist_ok=True,
        verbose=False,
    )

    collected = []
    t0 = time.perf_counter()
    for i, result in enumerate(results):
        collected.append(result)
        if (i + 1) % 50 == 0:
            print(f"      已处理 {i + 1} 帧 ...")
    wall = time.perf_counter() - t0

    speed = summarize_speed(collected, "drone.pt 视频推理")
    track = compute_tracking_metrics(collected)

    print("\n[追踪质量]")
    for key, value in track.items():
        if key != "note":
            print(f"  {key:<26}: {value}")
    wall_fps = len(collected) / wall if wall > 0 else 0.0
    print(f"  {'wall_clock_fps':<26}: {wall_fps:.2f} FPS（含读写视频）")
    print(f"  [说明] {track['note']}")

    metrics = {
        "model_file": assets.drone_weights.name,
        "model_size_mb": round(assets.drone_weights.stat().st_size / 1024 / 1024, 2),
        "model_classes": {"0": "Bird", "1": "Drone"},
        "source_video": video.name,
        "video_resolution": f"{width}x{height}",
        "video_fps": round(float(fps_src), 2),
        "video_frames": n_frames_src,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "tracker": args.tracker,
        "speed": speed,
        "wall_clock_fps": round(wall_fps, 2),
        "tracking": track,
    }
    save_json(args.output / "video_metrics.json", metrics)
    print(f"[保存] 标注视频 -> {save_dir}")


if __name__ == "__main__":
    main()
