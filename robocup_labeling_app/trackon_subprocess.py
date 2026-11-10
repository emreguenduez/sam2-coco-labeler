from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load_rgb_frame(path: str) -> np.ndarray:
    try:
        import imageio.v3 as iio

        frame = np.asarray(iio.imread(path))
    except Exception:
        from PIL import Image

        with Image.open(path) as image_handle:
            frame = np.asarray(image_handle.convert("RGB"))

    if frame.ndim == 2:
        frame = np.repeat(frame[:, :, None], 3, axis=2)
    if frame.shape[2] > 3:
        frame = frame[:, :, :3]
    return np.ascontiguousarray(frame, dtype=np.uint8)


def run_trackon(request: dict[str, object]) -> tuple[np.ndarray, np.ndarray, str, str]:
    trackon_root = Path(str(request["trackon_root"])).expanduser()
    if str(trackon_root) not in sys.path:
        sys.path.insert(0, str(trackon_root))

    import torch
    from model.trackon_predictor import Predictor
    from utils.train_utils import load_args_from_yaml

    checkpoint_path = Path(str(request["checkpoint_path"])).expanduser()
    config_value = request.get("config_path")
    config_path = Path(str(config_value)).expanduser() if config_value else None
    support_grid_size = max(0, int(request.get("support_grid_size", 0)))

    model_args = load_args_from_yaml(str(config_path)) if config_path else None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = Predictor(
        model_args,
        checkpoint_path=str(checkpoint_path),
        support_grid_size=support_grid_size,
    ).to(device).eval()

    frame_paths = [str(path) for path in request["frame_paths"]]
    query_specs = request["query_specs"]
    frames = [_load_rgb_frame(frame_path) for frame_path in frame_paths]
    frame_size = frames[0].shape[:2] if frames else None
    if any(frame.shape[:2] != frame_size for frame in frames):
        raise RuntimeError("Track-On requires all frames to have the same resolution.")

    if query_specs:
        queries = torch.tensor(query_specs, dtype=torch.float32)[None]
    else:
        queries = torch.zeros((1, 0, 3), dtype=torch.float32)
    video = torch.from_numpy(np.stack(frames, axis=0))
    video = video.permute(0, 3, 1, 2).contiguous().float()[None]
    video = video.to(device)
    queries = queries.to(device)

    with torch.inference_mode():
        tracks, visibility = predictor(video, queries=queries)

    tracks_np = tracks[0].detach().cpu().numpy().astype(np.float32)
    visibility_np = visibility[0].detach().cpu().numpy().astype(bool)
    return tracks_np, visibility_np, checkpoint_path.name, device


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Track-On inference for RoboCup Labeling.")
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    tracks, visibility, model_name, device = run_trackon(request)
    np.savez_compressed(
        args.output,
        tracks=tracks,
        visibility=visibility,
        model_name=np.array(model_name),
        device=np.array(device),
    )


if __name__ == "__main__":
    main()
