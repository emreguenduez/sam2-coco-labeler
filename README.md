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
- COCO import and export for finished object masks and point annotations
- Undo and redo support for annotation edits, imports, deletions, and task creation
- X-AnyLabeling-style layout with a left tool rail, centered canvas, and right sidebar

## Run

```bash
pip install PyQt6
python3 run_app.py
```

## Notes

- Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.tif`, `.tiff`, `.webp`
- SAM2 mode expects the local `SAM2-Plus` bundle and a compatible SAM2 checkpoint already available in the workspace
- COCO export writes object masks as RLE segmentations and points as point-style annotations in the same JSON file
- Useful shortcuts: `Ctrl+U` opens an image folder, left/right arrows change frame, `F` finishes the current SAM2 mask, `B` clears SAM2 prompts, and `Ctrl+A` selects all frames when the timeline panel is focused
- The copied `X-AnyLabeling` directory is kept as reference only and is not modified by this starter app
