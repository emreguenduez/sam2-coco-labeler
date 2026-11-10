# RoboCup Labeling

This folder now contains a lightweight desktop image-labeling starter app with
an interface inspired by X-AnyLabeling.

## Features

- Classic top menu bar with `File`, `Tools`, `Export`, `View`, and `Help`
- Open an image folder with `Ctrl+U`
- Automatically show the first image in the selected folder
- Navigate between images with the left and right arrow keys
- Point tool for placing keypoints directly on the image
- Unique point ids per image, with optional id visualization on the canvas
- SAM2 prompt mode with point and box prompts for a single object
- Immediate grey mask preview from SAM2 while prompts are being added
- Finish the current SAM2 preview as a labeled object with its own object id
- Incoming-scene autolabeling for objects or keypoints
- One-off autolabeling for a folder containing RGB video frames
- COCO import and export for finished object masks and point annotations
- Undo and redo support for annotation edits, imports, deletions, and task creation
- X-AnyLabeling-style layout with a left tool rail, centered canvas, and right sidebar

## Run

```bash
pip install PyQt6
python3 run_app.py
```

## macOS setup (Apple Silicon)

```bash
./setup_mac.sh      # once: creates .venv with PyTorch, PyQt6, SAM2 deps
./start_app.sh      # every time
```

- SAM2 runs on the Apple GPU (MPS) automatically and falls back to the CPU if an op is unsupported; force a device with `ROBOCUP_DEVICE=cpu|mps|cuda`.
- The image embedding is cached per image, so only the first SAM2 prompt on an image is slow.
- Phone photos: run `.venv/bin/python prepare_images.py RAW_DIR DATASET_DIR --tag <place_camera>` first. It converts HEIC to JPEG, bakes in the EXIF rotation and downsizes to 2048 px.
- Check an export with `.venv/bin/python visualize_coco.py DATASET_DIR/annotations_coco_objects.json`, which writes overlay images to `DATASET_DIR/visualized/`.

## Notes

- Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.tif`, `.tiff`, `.webp`
- SAM2 mode expects the local `SAM2-Plus` bundle and a compatible SAM2 checkpoint already available in the workspace
- Set `AUTOLABELING_MODE=keypoints` and `OPENROUTER_API_KEY` to autolabel keypoints from the first frame with OpenRouter, then propagate them through the video with the selected point tracker
- Keypoint autolabeling sends `robocup_labeling_app/keypoints.txt` directly as the OpenRouter prompt, can be pointed elsewhere with `ROBOCUP_KEYPOINTS_FILE`, and uses `google/gemini-3.6-flash` unless `ROBOCUP_KEYPOINT_MODEL` is set
- CoTracker keypoint propagation passes a full-image query grid together with the VLM keypoints; tune the grid spacing with `ROBOCUP_KEYPOINT_GRID_STEP` (default `32`). TAPIR, BootsTAPIR, and Track-On skip the full-image grid and track only local samples around each VLM keypoint to avoid excessive memory use
- Each auto-labeled keypoint is tracked with a local circle of tracker samples and collapsed back to one point by median; tune with `ROBOCUP_KEYPOINT_LOCAL_SAMPLE_RADIUS` (default `6`) and `ROBOCUP_KEYPOINT_LOCAL_SAMPLE_COUNT` (default `8`)
- CoTracker visibility filtering can be relaxed with `ROBOCUP_COTRACKER_VISIBILITY_THRESHOLD` (default `0.5` in this app) and the offline window length with `ROBOCUP_COTRACKER_WINDOW_LEN` (default `60`)
- Compare automatic keypoint tracking backends with `ROBOCUP_KEYPOINT_TRACKER=sampt` (default, alias for the app's SAM2 point propagation), `cotracker`, `tapir`, `bootstapir`, or `trackon`; manual point propagation offers the same options in the point tracker dropdown
- SAM-PT/SAM2 writes `coco_annotations_auto_keypoints_sam2.json`, CoTracker keeps writing `coco_annotations_auto_keypoints.json`, and TAPIR/BootsTAPIR/Track-On write `coco_annotations_auto_keypoints_tapir.json` / `coco_annotations_auto_keypoints_bootstapir.json` / `coco_annotations_auto_keypoints_trackon.json` for side-by-side comparison
- TAPIR and BootsTAPIR use the official TapNet implementation when installed; set `ROBOCUP_TAPNET_ROOT` to a local checkout, and optionally set `ROBOCUP_TAPIR_CHECKPOINT` or `ROBOCUP_BOOTSTAPIR_CHECKPOINT`
- TAPIR inference resizes frames to `ROBOCUP_TAPIR_RESOLUTION` (default `256` for TAPIR, `512` for BootsTAPIR), with optional `ROBOCUP_TAPIR_RESIZE_HEIGHT`, `ROBOCUP_TAPIR_RESIZE_WIDTH`, and `ROBOCUP_TAPIR_QUERY_CHUNK_SIZE` (default `32`)
- Track-On uses the official `track_on` implementation when installed; set `ROBOCUP_TRACKON_ROOT` to the local checkout, `ROBOCUP_TRACKON_CHECKPOINT` to a Track-On checkpoint, optionally set `ROBOCUP_TRACKON_CONFIG`, tune `ROBOCUP_TRACKON_SUPPORT_GRID_SIZE` (default `0`), and set `ROBOCUP_TRACKON_PYTHON` to run Track-On in a separate environment while keeping the app in its usual environment
- Use `Tools > Autolabel Frame Folder` to select a folder containing the RGB frames for one video and process it as a background task
- COCO export writes object masks as RLE segmentations and points as point-style annotations in the same JSON file
- Useful shortcuts: `Ctrl+U` opens an image folder, left/right arrows change frame, `F` finishes the current SAM2 mask, `B` clears SAM2 prompts, and `Ctrl+A` selects all frames when the timeline panel is focused

## Optional TAPIR / BootsTAPIR Setup

```bash
cd ~/robocup/robocuplabeling
git clone https://github.com/google-deepmind/tapnet.git
cd tapnet
python3 -m pip install -r requirements_inference.txt
python3 -m pip install .
mkdir -p checkpoints
wget -P checkpoints https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.npy
wget -P checkpoints https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.npy
export ROBOCUP_TAPNET_ROOT=$PWD
```

For GPU inference, install a JAX build that matches the machine's CUDA and cuDNN setup before running the app.

## Optional Track-On Setup

```bash
cd ~/robocup/robocuplabeling/track_on
mamba create -n track_on_r python=3.12
mamba activate track_on_r
mamba install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
pip install -r requirements.txt
mkdir -p checkpoints
wget -O checkpoints/track_on_r.pt "https://huggingface.co/gorkaydemir/track_on_r/resolve/main/track_on_r.pt?download=true"
export ROBOCUP_TRACKON_ROOT=$PWD
export ROBOCUP_TRACKON_CHECKPOINT=$PWD/checkpoints/track_on_r.pt
export ROBOCUP_TRACKON_PYTHON="$(which python)"
```

Track-On checkpoints do not include the DINOv3 backbone weights. Request access to `facebook/dinov3-vits16plus-pretrain-lvd1689m` on Hugging Face and run `huggingface-cli login` in the environment before first inference.

After setting these variables, switch back to the usual app environment before launching the labeling app. The app will spawn `ROBOCUP_TRACKON_PYTHON` only for Track-On inference.
