from __future__ import annotations

import copy
import json
import base64
import mimetypes
import os
import re
import sys
import threading
import urllib.error
import urllib.request
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue

import numpy as np
from PIL import Image, ImageDraw
from PyQt6.QtCore import QLineF, QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QCursor,
    QImage,
    QImageReader,
    QKeySequence,
    QPainter,
    QPen,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".webp",
}

OBJECT_COLORS = [
    QColor(65, 184, 131, 120),
    QColor(62, 146, 204, 120),
    QColor(239, 142, 74, 120),
    QColor(217, 90, 101, 120),
    QColor(165, 114, 221, 120),
]

VLMGIST_SETTINGS = {
    'logger_severity': None,
    'logger_name': "VLM-GIST",
    'message_process': True,
    'message_results': True,
    'include_image': False,
    'scene_description': {
        'skip': True,
        'message_process': True,
        'message_results': True,
        'chat_completions': {
            'logger_severity': "warn",
            'endpoint': "OpenRouter",
            'model': "~google/gemini-pro-latest",
            'reasoning_effort': "none"
        },
        'system_prompt_role': "system",
        'system_prompt': "You are a visual perception system that identifies and analyzes objects and other visible features in an image.",
        'image_prompt_role': "user",
        'image_prompt_detail': "high",
        'description_prompt_role': "user",
        'description_prompt':
            "Please provide a complete and detailed description of the image above. Your description must explicitly address all visible objects or features, "
            "even if they are small or in the background. Do not leave out or ignore anything. Your answer should consist of at least fifteen and no more than twenty-five sentences. "
            "If you are unsure about identifying an object, make a single educated guess instead of describing it as an unknown object or discussing different possibilities."
    },
    'structured_description': {
        'skip': False,
        'message_process': True,
        'message_results': False,
        'chat_completions': {
            'logger_severity': "warn",
            'endpoint': "OpenRouter",
            'model': "~google/gemini-pro-latest",
            'reasoning_effort': "none"
        },
        'use_scene_description': False,
        'system_prompt_role': "system",
        'system_prompt': "You are a visual perception system that identifies and analyzes objects and other visible features in an image.",
        'image_prompt_role': "user",
        'image_prompt_detail': "high",
        'description_prompt_role': "user",
        'description_prompt':
            "Provide a list in JSON format that contains each object (including furniture, persons, and animals) visible in the image above. "
            "Explicitly include each object instance without exception as an individual list element. "
            "Never group multiple instances that are clearly distinct from one another. "
            "Each list element must be a dictionary with the fields label, description, box_2d, point_2d, distance, weight, moving, and safety. "
            "The label of all humans must be person. "
            "The description must be a single short sentence (max. 10 words, starting with 'A' or 'An'), "
            "that differs from the other descriptions and summarizes the most important information about the type, color, and appearance of the object, "
            "allowing for a visual identification of the object without knowing any of the descriptions generated for the other objects. "
            "The key box_2d must contain a bounding box of the object in [y_min, x_min, y_max, x_max] format normalized to pixel coordinates from 0 to 1000. "
            "The key point_2d must contain a point on the object in [y, x] format normalized to pixel coordinates from 0 to 1000. "
            "The key distance must indicate the estimated distance of the object from the camera in centimeters as an integer. "
            "The key weight must indicate the estimated weight of the object in grams as an integer. "
            "The key moving must indicate how likely it is that the object is is going to move or be moved within the next three seconds, "
            "on a Likert scale from 1 to 5, where 1 is very unlikely, 2 is likely, 3 is ambiguous or unknown, 4 is likely, and 5 is very likely. "
            "The key safety must indicate how safe the object is for a ten year old child to interact with it without supervision on a Likert scale scale from 1 to 5, "
            "where 1 is very unsafe, 2 is unsafe, 3 is ambiguous or unknown, 4 is safe, and 5 is very safe. Make sure to not ignore any object do not forget any of the required keys.",
        'response_type': "json",
        'keys_required': ['label', 'description', 'box_2d', 'point_2d', 'distance', 'weight', 'moving', 'safety'],
        'keys_required_types': ['str', 'str', 'box_yxyx[int1000]', 'point_yx[int1000]', 'int', 'int', 'likert5', 'likert5'],
        'keys_optional': [],
        'keys_optional_types': []
    },
    'detection': {
        'skip': False,
        'message_process': True,
        'message_results': False,
        'extract_from_description': True,
        'prompt_key': "label",
        'mmgroundingdino': {
            'logger_severity': "warn",
            'message_results': False,
            'nms_iou': None
        }
    },
    'segmentation': {
        'skip': True,
        'message_process': True,
        'message_results': False,
        'track': False,
        'sam2_realtime': {
            'logger_severity': "warn"
        }
    },
    'batch': {
        'logger_severity': "warn",
        'size': 0,
        'style': "multiprocessing",
        'retry': 2
    },
    'retry': 2
}

@dataclass
class KeypointAnnotation:
    point_id: int
    x: float
    y: float


@dataclass
class SamPromptPoint:
    prompt_id: int
    x: float
    y: float


@dataclass
class SamPromptBox:
    prompt_id: int
    x1: float
    y1: float
    x2: float
    y2: float

    def normalized(self) -> SamPromptBox:
        return SamPromptBox(
            prompt_id=self.prompt_id,
            x1=min(self.x1, self.x2),
            y1=min(self.y1, self.y2),
            x2=max(self.x1, self.x2),
            y2=max(self.y1, self.y2),
        )

    def to_xyxy(self) -> np.ndarray:
        box = self.normalized()
        return np.array([box.x1, box.y1, box.x2, box.y2], dtype=np.float32)


@dataclass
class MaskObject:
    object_id: int
    label: str
    mask: np.ndarray
    bbox: tuple[int, int, int, int]


@dataclass
class TimelineObjectRow:
    object_key: str
    label: str
    object_id: int
    visibility: list[bool]


@dataclass
class TaskObjectSelection:
    object_key: str
    object_id: int
    label: str
    bbox: tuple[int, int, int, int]
    mask: np.ndarray


@dataclass
class TaskPointSelection:
    point_key: str
    point_id: int
    x: float
    y: float


@dataclass
class TaskFrameSelection:
    frame_index: int
    image_path: str
    objects: list[TaskObjectSelection] = field(default_factory=list)
    points: list[TaskPointSelection] = field(default_factory=list)


@dataclass
class PropagationTask:
    task_id: int
    start_index: int
    end_index: int
    object_keys: list[str] = field(default_factory=list)
    point_keys: list[str] = field(default_factory=list)
    frames: list[TaskFrameSelection] = field(default_factory=list)
    use_reverse_pass: bool = False
    point_tracker: str = "sam2"
    status: str = "Ready"
    progress: int = 0


@dataclass
class VocabularyClassSpec:
    label: str
    description: str


@dataclass
class OpenVocabularyDetectionRequest:
    scene_name: str
    scene_path: Path
    image_dir: Path
    classes_file: Path | None
    class_specs: list[VocabularyClassSpec] = field(default_factory=list)


@dataclass
class DetectedObjectBox:
    label: str
    bbox: tuple[float, float, float, float]
    description: str = ""
    score: float | None = None


@dataclass
class OpenVocabularyDetectionResult:
    detections: list[DetectedObjectBox] = field(default_factory=list)
    message: str = "No detections"


@dataclass
class DetectedKeypoint:
    label: str
    x: float
    y: float


@dataclass
class OpenRouterKeypointResult:
    keypoints: list[DetectedKeypoint] = field(default_factory=list)
    message: str = "No keypoints"


@dataclass
class SceneBackgroundTask:
    task_id: int
    request: OpenVocabularyDetectionRequest
    output_path: str
    autolabel_mode: str = "objects"
    use_reverse_pass: bool = False
    status: str = "Queued"
    progress: int = 0
    message: str = "Waiting to start"


@dataclass
class SceneTaskUpdate:
    task_id: int
    status: str | None = None
    progress: int | None = None
    message: str | None = None
    output_path: str | None = None


@dataclass
class AppStateSnapshot:
    point_annotations: dict[str, list[KeypointAnnotation]]
    sam2_prompt_points: dict[str, list[SamPromptPoint]]
    sam2_prompt_boxes: dict[str, SamPromptBox | None]
    sam2_preview_masks: dict[str, np.ndarray | None]
    object_annotations: dict[str, list[MaskObject]]
    propagation_tasks: list[PropagationTask]
    current_index: int
    timeline_panel_mode: str
    selected_object_time_window: tuple[int, int] | None
    selected_point_time_window: tuple[int, int] | None
    selected_timeline_objects: set[str]
    selected_timeline_points: set[str]


def _point_tracker_display_name(point_tracker: str) -> str:
    return "CoTracker" if point_tracker == "cotracker" else "SAM2"


class OpenVocabularyDetectorInterface:
    def __init__(self, settings: object | None = None) -> None:
        self.settings = settings if settings is not None else copy.deepcopy(VLMGIST_SETTINGS)
        self.client = None
        self.last_error: str | None = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        try:
            import nimbro_api
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)
            return

        try:
            self.client = nimbro_api.misc.VlmGist(self.settings)
            self.last_error = None
        except Exception as exc:  # noqa: BLE001
            self.client = None
            self.last_error = str(exc)

    def submit_request(
        self,
        request: OpenVocabularyDetectionRequest,
    ) -> OpenVocabularyDetectionResult:
        if self.client is None:
            return OpenVocabularyDetectionResult(
                detections=[],
                message=f"Detector client unavailable: {self.last_error or 'not initialized'}",
            )

        class_count = len(request.class_specs)
        labels = ", ".join(spec.label for spec in request.class_specs[:4])
        if class_count > 4:
            labels += f", +{class_count - 4} more"
        if not labels:
            labels = "no labels"

        first_image_path = next(
            (
                path
                for path in sorted(request.image_dir.iterdir())
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            None,
        )
        if first_image_path is None:
            return OpenVocabularyDetectionResult(
                detections=[],
                message=f"No supported image found in {request.image_dir}",
            )

        try:
            class_lines: list[str] = []
            for spec in request.class_specs:
                label = spec.label.strip()
                if not label:
                    continue
                description = spec.description.strip()
                if description and description != label:
                    class_lines.append(f"- {label}: {description}")
                else:
                    class_lines.append(f"- {label}")

            self.client.set_settings(settings=VLMGIST_SETTINGS)

            new_prompt = f"""Provide a list in JSON format that contains each object listed in the following classes.txt that is visible in the image above. {chr(10).join(class_lines)}\n This list contains a label and then a short description of the object that should be found, separated by a whitespace. Explicitly include each object instance without exception as an individual list element. Never group multiple instances that are clearly distinct from one another. Each list element must be a dictionary with the fields label, description, box_2d, point_2d, distance, weight, moving, and safety. Each object's label must be exactly as provided in classes.txt. The description must be a single short sentence (max. 10 words, starting with 'A' or 'An'), that differs from the other descriptions and summarizes the most important information about the type, color, and appearance of the object, allowing for a visual identification of the object without knowing any of the descriptions generated for the other objects. The key box_2d must contain a bounding box of the object in [y_min, x_min, y_max, x_max] format normalized to pixel coordinates from 0 to 1000. The key point_2d must contain a point on the object in [x, y] format normalized to pixel coordinates from 0 to 1000. The key distance must indicate the estimated distance of the object from the camera in centimeters as an integer. The key weight must indicate the estimated weight of the object in grams as an integer. The key moving must indicate how likely it is that the object is is going to move or be moved within the next three seconds, on a Likert scale from 1 to 5, where 1 is very unlikely, 2 is likely, 3 is ambiguous or unknown, 4 is likely, and 5 is very likely. The key safety must indicate how safe the object is for a ten year old child to interact with it without supervision on a Likert scale from 1 to 5, where 1 is very unsafe, 2 is unsafe, 3 is ambiguous or unknown, 4 is safe, and 5 is very safe. Make sure to not ignore any object do not forget any of the required keys."""
            settings = self.client.get_settings()
            settings["structured_description"]["description_prompt"] = new_prompt

            ok, msg = self.client.set_settings(settings=settings)
            if not ok:
                return OpenVocabularyDetectionResult(
                    detections=[],
                    message=f"Failed to update VLM-GIST settings: {msg}",
                )

            success, message, result = self.client.run(image=str(first_image_path))
            self.client.visualize(result, image=str(first_image_path), output_dir="/home/user/eichler/")

        except Exception as exc:  # noqa: BLE001
            return OpenVocabularyDetectionResult(
                detections=[],
                message=f"Detector run failed on {first_image_path.name}: {exc}",
            )

        with Image.open(first_image_path) as first_image:
            image_size = (first_image.width, first_image.height)

        detections = self._parse_client_result_to_detections(
            result,
            image_size=image_size,
        )
        result_summary = type(result).__name__ if result is not None else "None"
        return OpenVocabularyDetectionResult(
            detections=detections,
            message=(
                f"Detector ran on {first_image_path.name} for {request.scene_name} "
                f"with {class_count} class request(s): {labels} "
                f"(success={success}, message={message}, result={result_summary}, detections={len(detections)})"
            ),
        )

    def _parse_client_result_to_detections(
        self,
        result: object,
        *,
        image_size: tuple[int, int],
    ) -> list[DetectedObjectBox]:
        if not isinstance(result, dict):
            return []

        detection_block = result.get("detection")
        if isinstance(detection_block, dict):
            return self._parse_detection_data_entries(
                result,
                image_size=image_size,
            )

        return self._parse_structured_description_entries(
            result,
            image_size=image_size,
        )

    def _parse_detection_data_entries(
        self,
        result: dict[str, object],
        *,
        image_size: tuple[int, int],
    ) -> list[DetectedObjectBox]:
        detection_block = result.get("detection")
        structured_block = result.get("structured_description")
        if not isinstance(detection_block, dict):
            return []

        detection_items = detection_block.get("data")
        if not isinstance(detection_items, list):
            return []

        descriptions_by_prompt: dict[str, str] = {}
        structured_items = (
            structured_block.get("data")
            if isinstance(structured_block, dict)
            else None
        )
        if isinstance(structured_items, list):
            for item in structured_items:
                if not isinstance(item, dict):
                    continue
                prompt_value = item.get("description")
                if not isinstance(prompt_value, str) or not prompt_value.strip():
                    continue
                descriptions_by_prompt[prompt_value] = prompt_value

        parsed: list[DetectedObjectBox] = []
        for item in detection_items:
            if not isinstance(item, dict):
                continue
            box_value = item.get("box_xyxy")
            if not self._is_valid_box_value(box_value):
                continue
            label_value = item.get("prompt")
            label = str(label_value).strip() if label_value is not None else ""
            if not label:
                label = "object"
            score_value = item.get("confidence")
            score = (
                float(score_value)
                if isinstance(score_value, (int, float))
                else None
            )
            parsed.append(
                DetectedObjectBox(
                    label=label,
                    bbox=self._pixel_box_to_pixel_box(box_value),
                    description=descriptions_by_prompt.get(label, label),
                    score=score,
                )
            )
        return parsed

    def _parse_structured_description_entries(
        self,
        result: dict[str, object],
        *,
        image_size: tuple[int, int],
    ) -> list[DetectedObjectBox]:
        structured_block = result.get("structured_description")
        if not isinstance(structured_block, dict):
            return []

        structured_items = structured_block.get("data")
        if not isinstance(structured_items, list):
            return []

        parsed: list[DetectedObjectBox] = []
        for item in structured_items:
            if not isinstance(item, dict):
                continue
            box_value = item.get("box_2d")
            if not self._is_valid_box_value(box_value):
                continue
            label_value = item.get("label")
            label = str(label_value).strip() if label_value is not None else ""
            if not label:
                label = "object"
            description_value = item.get("description")
            description = (
                str(description_value).strip()
                if description_value is not None
                else label
            )
            parsed.append(
                DetectedObjectBox(
                    label=label,
                    bbox=self._pixel_box_to_pixel_box(box_value),
                    description=description or label,
                    score=None,
                )
            )
        return parsed

    def _is_valid_box_value(self, value: object) -> bool:
        return (
            isinstance(value, (list, tuple))
            and len(value) >= 4
            and all(isinstance(item, (int, float)) for item in value[:4])
        )

    def _pixel_box_to_pixel_box(
        self,
        value: list[object] | tuple[object, ...],
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = [float(item) for item in value[:4]]
        return x1, y1, x2, y2


def _mask_bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    positions = np.argwhere(mask)
    if positions.size == 0:
        return None
    y_min, x_min = positions.min(axis=0)
    y_max, x_max = positions.max(axis=0)
    return int(x_min), int(y_min), int(x_max), int(y_max)


def _prompt_box_from_detection(
    detection: DetectedObjectBox,
    *,
    image_width: int,
    image_height: int,
    prompt_id: int,
) -> SamPromptBox | None:
    x1, y1, x2, y2 = detection.bbox
    x_min = max(0.0, min(float(x1), float(x2)))
    y_min = max(0.0, min(float(y1), float(y2)))
    x_max = min(float(image_width - 1), max(float(x1), float(x2)))
    y_max = min(float(image_height - 1), max(float(y1), float(y2)))
    if x_max <= x_min or y_max <= y_min:
        return None
    return SamPromptBox(
        prompt_id=prompt_id,
        x1=x_min,
        y1=y_min,
        x2=x_max,
        y2=y_max,
    ).normalized()


def _mask_from_prompt_box(
    prompt_box: SamPromptBox,
    *,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    mask = np.zeros((image_height, image_width), dtype=bool)
    box = prompt_box.normalized()
    x1 = max(0, min(int(round(box.x1)), image_width - 1))
    y1 = max(0, min(int(round(box.y1)), image_height - 1))
    x2 = max(0, min(int(round(box.x2)), image_width - 1))
    y2 = max(0, min(int(round(box.y2)), image_height - 1))
    if x2 < x1 or y2 < y1:
        return mask
    mask[y1:y2 + 1, x1:x2 + 1] = True
    return mask


def _sample_grid_points_from_mask(
    mask: np.ndarray | None,
    *,
    step: int,
) -> list[tuple[float, float]]:
    if mask is None:
        return []
    mask_array = np.asarray(mask).astype(bool)
    if mask_array.ndim != 2 or not np.any(mask_array):
        return []

    y_indices, x_indices = np.where(mask_array)
    y_min = int(y_indices.min())
    y_max = int(y_indices.max())
    x_min = int(x_indices.min())
    x_max = int(x_indices.max())
    grid_step = max(1, int(step))

    sampled_points: list[tuple[float, float]] = []
    y_start = y_min + grid_step // 2
    x_start = x_min + grid_step // 2
    for y in range(y_start, y_max + 1, grid_step):
        for x in range(x_start, x_max + 1, grid_step):
            if 0 <= y < mask_array.shape[0] and 0 <= x < mask_array.shape[1] and mask_array[y, x]:
                sampled_points.append((float(x), float(y)))

    if sampled_points:
        return sampled_points

    return [(float(np.mean(x_indices)), float(np.mean(y_indices)))]


def _mean_keypoint_prompt(points: list[KeypointAnnotation]) -> SamPromptPoint | None:
    if not points:
        return None
    return SamPromptPoint(
        prompt_id=1,
        x=float(sum(point.x for point in points) / len(points)),
        y=float(sum(point.y for point in points) / len(points)),
    )


def _object_timeline_key_static(obj: MaskObject) -> str:
    return f"#{obj.object_id} {obj.label}"


def _image_export_name_static(image_path: Path, folder_path: Path | None) -> str:
    if folder_path is None:
        return image_path.name
    try:
        return str(image_path.relative_to(folder_path))
    except ValueError:
        return image_path.name


def _encode_mask_to_rle_static(mask: np.ndarray) -> dict[str, object]:
    flat_mask = mask.astype(np.uint8).flatten(order="F")
    counts: list[int] = []
    current_value = 0
    run_length = 0
    for value in flat_mask:
        if int(value) == current_value:
            run_length += 1
            continue
        counts.append(run_length)
        run_length = 1
        current_value = int(value)
    counts.append(run_length)
    return {
        "size": [int(mask.shape[0]), int(mask.shape[1])],
        "counts": counts,
    }


def _bbox_to_coco_static(
    bbox: tuple[int, int, int, int],
) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [
        float(x1),
        float(y1),
        float(x2 - x1 + 1),
        float(y2 - y1 + 1),
    ]


def _build_coco_payload_for_scene(
    *,
    image_paths: list[Path],
    folder_path: Path,
    object_annotations: dict[str, list[MaskObject]] | None = None,
    point_annotations: dict[str, list[KeypointAnnotation]] | None = None,
) -> dict[str, object]:
    category_ids: dict[str, int] = {}
    categories: list[dict[str, object]] = []
    images_payload: list[dict[str, object]] = []
    annotations_payload: list[dict[str, object]] = []
    annotation_id = 1
    point_category_id: int | None = None

    def ensure_category(label: str) -> int:
        category_id = category_ids.get(label)
        if category_id is not None:
            return category_id
        category_id = len(category_ids) + 1
        category_ids[label] = category_id
        categories.append(
            {
                "id": category_id,
                "name": label,
                "supercategory": "object",
            }
        )
        return category_id

    for image_id, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as image_handle:
            width, height = image_handle.width, image_handle.height
        images_payload.append(
            {
                "id": image_id,
                "file_name": _image_export_name_static(image_path, folder_path),
                "width": width,
                "height": height,
            }
        )
        image_key = str(image_path)
        if object_annotations is not None:
            for obj in object_annotations.get(image_key, []):
                category_id = ensure_category(obj.label)
                annotations_payload.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "annotation_type": "object",
                        "object_id": obj.object_id,
                        "label": obj.label,
                        "bbox": _bbox_to_coco_static(obj.bbox),
                        "segmentation": _encode_mask_to_rle_static(obj.mask),
                        "area": int(np.count_nonzero(obj.mask)),
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1
        if point_annotations is not None:
            if point_category_id is None:
                point_category_id = ensure_category("keypoint")
                categories[-1]["supercategory"] = "point"
            for point in point_annotations.get(image_key, []):
                annotations_payload.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": point_category_id,
                        "annotation_type": "point",
                        "point_id": point.point_id,
                        "point": [round(point.x, 3), round(point.y, 3)],
                        "keypoints": [round(point.x, 3), round(point.y, 3), 2],
                        "num_keypoints": 1,
                        "bbox": [
                            round(point.x - 2.0, 3),
                            round(point.y - 2.0, 3),
                            4.0,
                            4.0,
                        ],
                        "area": 0,
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1

    return {
        "info": {
            "description": "RoboCup Labeling COCO export",
            "version": "1.0",
        },
        "images": images_payload,
        "annotations": annotations_payload,
        "categories": categories,
    }


def _build_vlmgist_settings_for_request(
    request: OpenVocabularyDetectionRequest,
) -> dict[str, object]:
    
     return VLMGIST_SETTINGS


def _current_autolabeling_mode() -> str:
    return "tshirt" if os.environ.get("AUTOLABELING_MODE", "").strip().lower() == "tshirt" else "objects"


TSHIRT_KEYPOINT_ID_BY_LABEL: dict[str, int] = {
    "hem left": 1,
    "armpit left": 2,
    "sleeve left bottom": 3,
    "sleeve left top": 4,
    "shoulder left": 5,
    "neckline left": 6,
    "neckline right": 7,
    "shoulder right": 8,
    "sleeve right top": 9,
    "sleeve right bottom": 10,
    "armpit right": 11,
    "hem right": 12,
}


def _normalize_tshirt_keypoint_label(label: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", label.strip().lower())
    return " ".join(cleaned.split())


def _select_tshirt_seed_frame_indices(frame_count: int) -> list[int]:
    if frame_count <= 0:
        return []
    quarter_indices = [
        min(frame_count - 1, max(0, (frame_count * 1) // 4)),
        min(frame_count - 1, max(0, (frame_count * 2) // 4)),
        min(frame_count - 1, max(0, (frame_count * 3) // 4)),
    ]
    deduplicated: list[int] = []
    for index in quarter_indices:
        if index not in deduplicated:
            deduplicated.append(index)
    return deduplicated


class OpenRouterKeypointExtractor:
    def __init__(self) -> None:
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        self.model = os.environ.get("ROBOCUP_TSHIRT_KEYPOINT_MODEL", "google/gemini-2.5-flash")
        tshirt_keypoint_lines = [
            f"{point_id}: {label}"
            for label, point_id in TSHIRT_KEYPOINT_ID_BY_LABEL.items()
        ]
        self.prompt = os.environ.get(
            "ROBOCUP_TSHIRT_KEYPOINT_PROMPT",
            (
                "Return a JSON array containing all requested tshirt keypoints visible in the image. "
                "Only use the following exact tshirt keypoints and labels:\n"
                f"{chr(10).join(tshirt_keypoint_lines)}\n"
                "Each array element must describe exactly one visible keypoint and must contain the fields "
                "'id', 'label', and 'point_2d'. "
                "The 'id' must match the mapping above. "
                "The 'label' must exactly match one of the labels above. "
                "The field 'point_2d' must be the keypoint location as normalized [y, x] coordinates from 0 to 1000. "
                "Do not invent additional labels, do not rename labels, and do not include invisible keypoints."
            ),
        )
        self.point_order = os.environ.get("ROBOCUP_TSHIRT_KEYPOINT_ORDER", "yx").strip().lower()

    def extract_keypoints(self, image_path: Path) -> OpenRouterKeypointResult:
        if not self.api_key:
            return OpenRouterKeypointResult(
                keypoints=[],
                message="OPENROUTER_API_KEY is not set.",
            )

        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_b64}",
                            },
                        },
                    ],
                }
            ],
        }

        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            return OpenRouterKeypointResult([], f"OpenRouter HTTP error {exc.code}: {detail}")
        except Exception as exc:  # noqa: BLE001
            return OpenRouterKeypointResult([], f"OpenRouter request failed: {exc}")

        try:
            content = response_payload["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            return OpenRouterKeypointResult([], f"OpenRouter response format error: {exc}")

        text_content = self._content_to_text(content)
        raw_items = self._extract_json_payload(text_content)
        if raw_items is None:
            return OpenRouterKeypointResult([], "OpenRouter did not return parseable JSON keypoints.")

        with Image.open(image_path) as image_handle:
            image_width, image_height = image_handle.width, image_handle.height

        keypoints: list[DetectedKeypoint] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            label_value = item.get("label")
            label = str(label_value).strip() if label_value is not None else ""
            point_value = item.get("point_2d", item.get("point"))
            if not label or not isinstance(point_value, (list, tuple)) or len(point_value) < 2:
                continue
            try:
                x_value, y_value = self._point_to_pixels(
                    point_value,
                    width=image_width,
                    height=image_height,
                )
            except Exception:
                continue
            keypoints.append(DetectedKeypoint(label=label, x=x_value, y=y_value))

        return OpenRouterKeypointResult(
            keypoints=keypoints,
            message=f"Extracted {len(keypoints)} keypoint(s) from {image_path.name}.",
        )

    def _content_to_text(self, content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            return "\n".join(part for part in parts if part)
        return str(content)

    def _extract_json_payload(self, text: str) -> list[object] | None:
        stripped = text.strip()
        candidates = [stripped]
        fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)```", stripped, flags=re.IGNORECASE | re.DOTALL)
        candidates.extend(block.strip() for block in fenced_blocks if block.strip())
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                if isinstance(parsed.get("keypoints"), list):
                    return parsed["keypoints"]
                return [parsed]
        return None

    def _point_to_pixels(
        self,
        point_value: list[object] | tuple[object, ...],
        *,
        width: int,
        height: int,
    ) -> tuple[float, float]:
        first = float(point_value[0])
        second = float(point_value[1])
        if self.point_order == "yx":
            x_raw, y_raw = second, first
        else:
            x_raw, y_raw = first, second

        if 0.0 <= x_raw <= 1.0 and 0.0 <= y_raw <= 1.0:
            return x_raw * width, y_raw * height
        if 0.0 <= x_raw <= 1000.0 and 0.0 <= y_raw <= 1000.0:
            return x_raw / 1000.0 * width, y_raw / 1000.0 * height
        return x_raw, y_raw


def _run_scene_processing_pipeline(
    task: SceneBackgroundTask,
    *,
    workspace_root: Path,
    package_root: Path,
    progress_callback,
) -> tuple[str, Path]:
    progress_callback(5, "Initializing detector")
    detector = OpenVocabularyDetectorInterface(
        settings=_build_vlmgist_settings_for_request(task.request),
    )
    detection_result = detector.submit_request(task.request)

    image_paths = sorted(
        path
        for path in task.request.image_dir.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise RuntimeError("The scene folder does not contain supported image files.")

    first_image_path = image_paths[0]
    with Image.open(first_image_path) as first_image:
        image_width, image_height = first_image.width, first_image.height

    progress_callback(20, "Seeding objects from detector result")
    image_engine = Sam2ImageEngine(
        workspace_root=workspace_root,
        package_root=package_root,
    )
    seeded_objects: list[MaskObject] = []
    for detection_index, detection in enumerate(detection_result.detections, start=1):
        label = detection.label.strip()
        if not label:
            continue
        prompt_box = _prompt_box_from_detection(
            detection,
            image_width=image_width,
            image_height=image_height,
            prompt_id=detection_index,
        )
        if prompt_box is None:
            continue

        preview_mask: np.ndarray | None = None
        try:
            preview_mask, _status = image_engine.predict(
                first_image_path,
                [],
                prompt_box,
            )
        except Exception:
            preview_mask = None

        if preview_mask is None or not np.any(preview_mask):
            preview_mask = _mask_from_prompt_box(
                prompt_box,
                image_width=image_width,
                image_height=image_height,
            )

        bbox = _mask_bbox_from_mask(preview_mask)
        if bbox is None:
            continue

        seeded_objects.append(
            MaskObject(
                object_id=len(seeded_objects) + 1,
                label=label,
                mask=preview_mask.copy(),
                bbox=bbox,
            )
        )

    object_annotations: dict[str, list[MaskObject]] = {
        str(path): [] for path in image_paths
    }
    if seeded_objects:
        object_annotations[str(first_image_path)] = copy.deepcopy(seeded_objects)

    if seeded_objects:
        progress_callback(35, "Running propagation")
        propagation_task = PropagationTask(
            task_id=task.task_id,
            start_index=0,
            end_index=len(image_paths) - 1,
            object_keys=[
                _object_timeline_key_static(obj) for obj in seeded_objects
            ],
            point_keys=[],
            frames=[
                TaskFrameSelection(
                    frame_index=frame_index,
                    image_path=str(image_path),
                    objects=(
                        [
                            TaskObjectSelection(
                                object_key=_object_timeline_key_static(obj),
                                object_id=obj.object_id,
                                label=obj.label,
                                bbox=obj.bbox,
                                mask=obj.mask.copy(),
                            )
                            for obj in seeded_objects
                        ]
                        if frame_index == 0
                        else []
                    ),
                    points=[],
                )
                for frame_index, image_path in enumerate(image_paths)
            ],
            use_reverse_pass=task.use_reverse_pass,
        )
        video_engine = Sam2VideoPropagationEngine(
            workspace_root=workspace_root,
            package_root=package_root,
        )
        object_results, _point_results, propagation_status = video_engine.run_task(
            folder_path=task.request.image_dir,
            frame_paths=image_paths,
            task=propagation_task,
            progress_callback=lambda current, total: progress_callback(
                35 + int((current / total) * 55) if total > 0 else 35,
                "Running propagation",
            ),
        )
        for frame_index, image_path in enumerate(image_paths):
            propagated_objects = copy.deepcopy(object_results.get(frame_index, []))
            if frame_index == 0 and not propagated_objects:
                propagated_objects = copy.deepcopy(seeded_objects)
            object_annotations[str(image_path)] = propagated_objects
    else:
        propagation_status = "No detected objects to propagate"

    progress_callback(94, "Saving output annotations")
    output_path = Path(task.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _build_coco_payload_for_scene(
        image_paths=image_paths,
        folder_path=task.request.image_dir,
        object_annotations=object_annotations,
    )
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    progress_callback(100, "Finished")
    seeded_count = len(seeded_objects)
    return (
        f"{detection_result.message} | seeded {seeded_count} object(s) | {propagation_status}",
        output_path,
    )


def _run_tshirt_scene_processing_pipeline(
    task: SceneBackgroundTask,
    *,
    workspace_root: Path,
    package_root: Path,
    progress_callback,
) -> tuple[str, Path]:
    image_paths = sorted(
        path
        for path in task.request.image_dir.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise RuntimeError("The scene folder does not contain supported image files.")

    seed_indices = _select_tshirt_seed_frame_indices(len(image_paths))
    if not seed_indices:
        raise RuntimeError("Could not determine tshirt keypoint seed frames.")

    progress_callback(5, "Initializing OpenRouter keypoint extractor")
    extractor = OpenRouterKeypointExtractor()
    point_annotations: dict[str, list[KeypointAnnotation]] = {
        str(path): [] for path in image_paths
    }
    extraction_messages: list[str] = []
    unknown_labels: set[str] = set()

    for seed_number, frame_index in enumerate(seed_indices, start=1):
        frame_path = image_paths[frame_index]
        progress_callback(
            10 + int((seed_number / max(len(seed_indices), 1)) * 20),
            f"Extracting keypoints from frame {frame_index + 1}",
        )
        extraction_result = extractor.extract_keypoints(frame_path)
        extraction_messages.append(extraction_result.message)
        frame_points_by_id: dict[int, KeypointAnnotation] = {}
        for detection in extraction_result.keypoints:
            normalized_label = _normalize_tshirt_keypoint_label(detection.label)
            point_id = TSHIRT_KEYPOINT_ID_BY_LABEL.get(normalized_label)
            if point_id is None:
                unknown_labels.add(detection.label.strip() or detection.label)
                continue
            frame_points_by_id[point_id] = KeypointAnnotation(
                point_id=point_id,
                x=detection.x,
                y=detection.y,
            )
        point_annotations[str(frame_path)] = list(frame_points_by_id.values())

    point_keys = sorted(
        {
            f"Point {point.point_id}"
            for annotations in point_annotations.values()
            for point in annotations
        },
        key=lambda point_key: int(point_key.split(" ", 1)[1]),
    )
    if point_keys:
        progress_callback(30, "Initializing SAM2 and CoTracker")
        sam2_engine = Sam2ImageEngine(
            workspace_root=workspace_root,
            package_root=package_root,
        )
        video_engine = Sam2VideoPropagationEngine(
            workspace_root=workspace_root,
            package_root=package_root,
        )
        grid_step = max(1, int(os.environ.get("ROBOCUP_TSHIRT_GRID_STEP", "12")))
        mask_messages: list[str] = []
        query_specs: list[tuple[int, float, float]] = []
        keypoint_track_assignments: list[tuple[int, int, int]] = []

        for seed_number, frame_index in enumerate(seed_indices, start=1):
            frame_path = image_paths[frame_index]
            seed_points = point_annotations.get(str(frame_path), [])
            if not seed_points:
                continue

            progress_callback(
                35 + int((seed_number / max(len(seed_indices), 1)) * 20),
                f"Segmenting tshirt on frame {frame_index + 1}",
            )
            mean_prompt = _mean_keypoint_prompt(seed_points)
            tshirt_mask = None
            mask_status = "No keypoints available for SAM2 prompt"
            if mean_prompt is not None:
                tshirt_mask, mask_status = sam2_engine.predict(
                    frame_path,
                    [mean_prompt],
                    None,
                )
            mask_messages.append(f"{frame_path.name}: {mask_status}")

            sampled_grid_points = _sample_grid_points_from_mask(
                tshirt_mask,
                step=grid_step,
            )
            if not sampled_grid_points:
                sampled_grid_points = [
                    (float(point.x), float(point.y))
                    for point in seed_points
                ]

            frame_query_indices: list[int] = []
            for x_value, y_value in sampled_grid_points:
                frame_query_indices.append(len(query_specs))
                query_specs.append((frame_index, float(x_value), float(y_value)))

            for point in seed_points:
                if not frame_query_indices:
                    continue
                best_query_index = min(
                    frame_query_indices,
                    key=lambda query_index: (query_specs[query_index][1] - point.x) ** 2
                    + (query_specs[query_index][2] - point.y) ** 2,
                )
                keypoint_track_assignments.append(
                    (point.point_id, frame_index, best_query_index)
                )

        if query_specs and keypoint_track_assignments:
            progress_callback(60, "Running mask-guided CoTracker propagation")
            pred_tracks, visibility_mask = video_engine._run_cotracker_queries(
                frame_paths=image_paths,
                query_specs=query_specs,
                backward_tracking=True,
            )

            propagation_status = (
                f"points: SAM2 mask seeding + {video_engine.cotracker_model_name} "
                f"on {video_engine.cotracker_device}"
            )
            for frame_index, image_path in enumerate(image_paths):
                best_predictions: dict[int, tuple[int, bool, float, float]] = {}
                for point_id, seed_frame_idx, query_index in keypoint_track_assignments:
                    xy = pred_tracks[frame_index, query_index]
                    frame_distance = abs(frame_index - seed_frame_idx)
                    is_visible = bool(visibility_mask[frame_index, query_index])
                    current_best = best_predictions.get(point_id)
                    if current_best is not None:
                        current_distance, current_visible, _current_x, _current_y = current_best
                        if current_distance < frame_distance:
                            continue
                        if current_distance == frame_distance and current_visible and not is_visible:
                            continue
                    best_predictions[point_id] = (
                        frame_distance,
                        is_visible,
                        float(xy[0]),
                        float(xy[1]),
                    )

                point_annotations[str(image_path)] = [
                    KeypointAnnotation(point_id=point_id, x=x_value, y=y_value)
                    for point_id, (_distance, _visible, x_value, y_value) in sorted(
                        best_predictions.items()
                    )
                ]
        else:
            propagation_status = "No valid SAM2 mask/grid seeds available for propagation"
    else:
        propagation_status = "No keypoints extracted to propagate"
        mask_messages = []

    progress_callback(94, "Saving output annotations")
    output_path = Path(task.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _build_coco_payload_for_scene(
        image_paths=image_paths,
        folder_path=task.request.image_dir,
        object_annotations=None,
        point_annotations=point_annotations,
    )
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    progress_callback(100, "Finished")
    unknown_labels_text = ""
    if unknown_labels:
        unknown_labels_text = f" | ignored unknown labels: {', '.join(sorted(unknown_labels))}"
    mask_summary = ""
    if point_keys and mask_messages:
        mask_summary = f" | mask-seeded grid propagation from {len(mask_messages)} frame(s)"
    return (
        f"{' | '.join(extraction_messages)} | seeded {len(point_keys)} keypoint id(s){unknown_labels_text}{mask_summary} | {propagation_status}",
        output_path,
    )


class Sam2ImageEngine:
    def __init__(self, workspace_root: Path, package_root: Path) -> None:
        self.workspace_root = workspace_root
        self.package_root = package_root
        self.predictor = None
        self.device = None
        self.torch = None
        self.model_name = "SAM2 not loaded"
        self.last_error: str | None = None

    def predict(
        self,
        image_path: Path,
        prompt_points: list[SamPromptPoint],
        prompt_box: SamPromptBox | None,
    ) -> tuple[np.ndarray | None, str]:
        if not prompt_points and prompt_box is None:
            return None, "No prompts"

        self._ensure_loaded()

        image = np.array(Image.open(image_path).convert("RGB"))
        with self.torch.inference_mode():
            with self._autocast_context():
                self.predictor.set_image(image)

                point_coords = None
                point_labels = None
                if prompt_points:
                    point_coords = np.array(
                        [[point.x, point.y] for point in prompt_points],
                        dtype=np.float32,
                    )
                    point_labels = np.ones(len(prompt_points), dtype=np.int32)

                box = prompt_box.to_xyxy() if prompt_box is not None else None
                masks, scores, _ = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box,
                    multimask_output=False,
                )

        mask = masks[0] if getattr(masks, "ndim", 0) == 3 else masks
        score = float(scores[0]) if np.ndim(scores) else float(scores)
        return np.asarray(mask).astype(bool), f"{self.model_name} on {self.device} | score {score:.3f}"

    def _ensure_loaded(self) -> None:
        if self.predictor is not None:
            return

        if str(self.package_root) not in sys.path:
            sys.path.insert(0, str(self.package_root))

        import torch
        from hydra import initialize_config_module
        from hydra.core.global_hydra import GlobalHydra
        import sam2  # noqa: F401
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        # The local sam2_plus package clears Hydra and rebinds the config root to
        # `sam2_plus`. For image prompting we build the standard `sam2` image model,
        # so we explicitly restore Hydra's config search path to the `sam2` module.
        GlobalHydra.instance().clear()
        initialize_config_module("sam2", version_base="1.2")

        config_name, checkpoint_path, model_name = self._discover_model_files()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._configure_torch_runtime(torch, device)
        model = build_sam2(config_name, str(checkpoint_path), device=device)
        self.predictor = SAM2ImagePredictor(model)
        self.torch = torch
        self.device = device
        self.model_name = model_name
        self.last_error = None

    def _autocast_context(self):
        if self.device == "cuda" and self.torch is not None:
            return self.torch.autocast(device_type="cuda", dtype=self.torch.bfloat16)
        return nullcontext()

    def _configure_torch_runtime(self, torch, device: str) -> None:
        if device != "cuda":
            return
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def _discover_model_files(self) -> tuple[str, Path, str]:
        env_ckpt = os.environ.get("ROBOCUP_SAM2_CHECKPOINT")
        env_cfg = os.environ.get("ROBOCUP_SAM2_CONFIG")
        if env_ckpt and env_cfg:
            ckpt_path = Path(env_ckpt)
            if ckpt_path.exists():
                return env_cfg, ckpt_path, ckpt_path.name

        candidates = [
            (
                "configs/sam2.1/sam2.1_hiera_b+.yaml",
                "sam2.1_hiera_base_plus.pt",
                [
                    self.workspace_root / "groundedsam2/checkpoints/sam2.1_hiera_base_plus.pt",
                    self.workspace_root / "ros2_ws/src/sam2_ros/data/sam2.1_hiera_base_plus.pt",
                ],
            ),
            (
                "configs/sam2.1/sam2.1_hiera_s.yaml",
                "sam2.1_hiera_small.pt",
                [
                    self.workspace_root / "groundedsam2/checkpoints/sam2.1_hiera_small.pt",
                    self.workspace_root / "ros2_ws/src/sam2_ros/data/sam2.1_hiera_small.pt",
                ],
            ),
            (
                "configs/sam2.1/sam2.1_hiera_t.yaml",
                "sam2.1_hiera_tiny.pt",
                [
                    self.workspace_root / "groundedsam2/checkpoints/sam2.1_hiera_tiny.pt",
                    self.workspace_root / "ros2_ws/src/sam2_ros/data/sam2.1_hiera_tiny.pt",
                ],
            ),
            (
                "configs/sam2.1/sam2.1_hiera_l.yaml",
                "sam2.1_hiera_large.pt",
                [
                    self.workspace_root / "groundedsam2/checkpoints/sam2.1_hiera_large.pt",
                    self.workspace_root / "ros2_ws/src/sam2_ros/data/sam2.1_hiera_large.pt",
                ],
            ),
        ]

        for config_name, model_name, paths in candidates:
            for path in paths:
                if path.exists():
                    return config_name, path, model_name

        raise RuntimeError(
            "No SAM2 checkpoint was found. Set ROBOCUP_SAM2_CHECKPOINT and ROBOCUP_SAM2_CONFIG or place a checkpoint in the workspace."
        )


class Sam2VideoPropagationEngine:
    def __init__(self, workspace_root: Path, package_root: Path) -> None:
        self.workspace_root = workspace_root
        self.package_root = package_root
        self.mask_predictor = None
        self.point_predictor = None
        self.cotracker_predictor = None
        self.torch = None
        self.device = None
        self.model_name = "SAM2-Plus not loaded"
        self.cotracker_model_name = "CoTracker not loaded"
        self.cotracker_device = None
        self.last_error: str | None = None
        self.cotracker_last_error: str | None = None
        self.score_thresh = 0.0
        self.point_radius = 50
        self.point_sigma = 16

    def run_task(
        self,
        folder_path: Path,
        frame_paths: list[Path],
        task: PropagationTask,
        progress_callback=None,
    ) -> tuple[dict[int, list[MaskObject]], dict[int, list[KeypointAnnotation]], str]:
        if not frame_paths:
            return {}, {}, "No frames"

        frame_names = [frame_path.name for frame_path in frame_paths]
        ordered_frames = list(task.frames)
        object_results: dict[int, list[MaskObject]] = {}
        point_results: dict[int, list[KeypointAnnotation]] = {}
        object_stage_count = int(bool(task.object_keys)) * (2 if task.use_reverse_pass else 1)
        if task.point_keys:
            point_stage_count = 1 if task.point_tracker == "cotracker" else (2 if task.use_reverse_pass else 1)
        else:
            point_stage_count = 0
        stage_count = object_stage_count + point_stage_count
        total_steps = max(len(task.frames) * max(stage_count, 1), 1)
        completed_steps = 0

        if progress_callback is not None:
            progress_callback(0, total_steps)

        forward_object_results: dict[int, list[MaskObject]] = {}
        reverse_object_results: dict[int, list[MaskObject]] = {}
        forward_point_results: dict[int, list[KeypointAnnotation]] = {}
        reverse_point_results: dict[int, list[KeypointAnnotation]] = {}
        status_parts: list[str] = []

        if task.object_keys:
            self._ensure_loaded()
            forward_object_results, completed_steps = self._run_mask_propagation(
                folder_path=folder_path,
                frame_names=frame_names,
                ordered_frames=ordered_frames,
                completed_steps=completed_steps,
                total_steps=total_steps,
                progress_callback=progress_callback,
            )
            if task.use_reverse_pass:
                reverse_object_results, completed_steps = self._run_mask_propagation(
                    folder_path=folder_path,
                    frame_names=list(reversed(frame_names)),
                    ordered_frames=list(reversed(ordered_frames)),
                    completed_steps=completed_steps,
                    total_steps=total_steps,
                    progress_callback=progress_callback,
                )
            object_results = self._merge_object_results(
                task=task,
                forward_results=forward_object_results,
                reverse_results=reverse_object_results,
            )
            status_parts.append(f"objects: {self.model_name} on {self.device}")
        if task.point_keys:
            if task.point_tracker == "cotracker":
                forward_point_results, completed_steps = self._run_cotracker_point_propagation(
                    frame_paths=frame_paths,
                    ordered_frames=ordered_frames,
                    backward_tracking=task.use_reverse_pass,
                    completed_steps=completed_steps,
                    total_steps=total_steps,
                    progress_callback=progress_callback,
                )
                point_results = forward_point_results
                status_parts.append(
                    f"points: {self.cotracker_model_name} on {self.cotracker_device}"
                )
            else:
                self._ensure_loaded()
                forward_point_results, completed_steps = self._run_point_propagation(
                    folder_path=folder_path,
                    frame_names=frame_names,
                    ordered_frames=ordered_frames,
                    completed_steps=completed_steps,
                    total_steps=total_steps,
                    progress_callback=progress_callback,
                )
                if task.use_reverse_pass:
                    reverse_point_results, completed_steps = self._run_point_propagation(
                        folder_path=folder_path,
                        frame_names=list(reversed(frame_names)),
                        ordered_frames=list(reversed(ordered_frames)),
                        completed_steps=completed_steps,
                        total_steps=total_steps,
                        progress_callback=progress_callback,
                    )
                point_results = self._merge_point_results(
                    task=task,
                    forward_results=forward_point_results,
                    reverse_results=reverse_point_results,
                )
                status_parts.append(f"points: {self.model_name} on {self.device}")

        if progress_callback is not None:
            progress_callback(total_steps, total_steps)

        return object_results, point_results, " | ".join(status_parts) or "No propagation"

    def _ensure_loaded(self) -> None:
        if self.mask_predictor is not None and self.point_predictor is not None:
            return

        if str(self.package_root) not in sys.path:
            sys.path.insert(0, str(self.package_root))

        try:
            import torch
            from hydra import initialize_config_module
            from hydra.core.global_hydra import GlobalHydra
            import sam2_plus  # noqa: F401
            from sam2_plus.build_sam import build_sam2_video_predictor_plus
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)
            raise RuntimeError(
                "Could not import the local SAM2-Plus bundle. Make sure SAM2-Plus dependencies are installed."
            ) from exc

        # The standard SAM2 image engine resets Hydra to the `sam2` config module.
        # Before building SAM2-Plus video predictors we must explicitly restore the
        # config search path to `sam2_plus`, otherwise Hydra can fail while locating
        # the predictor target or composing the plus config.
        GlobalHydra.instance().clear()
        initialize_config_module("sam2_plus", version_base="1.2")

        config_name, checkpoint_path, model_name = self._discover_model_files()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._configure_torch_runtime(torch, device)
        common_kwargs = {
            "config_file": config_name,
            "ckpt_path": str(checkpoint_path),
            "device": device,
            "apply_postprocessing": False,
            "hydra_overrides_extra": [
                "++model.non_overlap_masks=false",
            ],
            "vos_optimized": False,
        }
        self.mask_predictor = build_sam2_video_predictor_plus(
            task="mask",
            **common_kwargs,
        )
        self.point_predictor = build_sam2_video_predictor_plus(
            task="point",
            **common_kwargs,
        )
        self.torch = torch
        self.device = device
        self.model_name = model_name
        self.last_error = None

    def _discover_model_files(self) -> tuple[str, Path, str]:
        env_ckpt = os.environ.get("ROBOCUP_SAM2_PLUS_CHECKPOINT")
        env_cfg = os.environ.get("ROBOCUP_SAM2_PLUS_CONFIG")
        default_config = "configs/sam2.1/sam2.1_hiera_b+_predmasks_decoupled_MAME.yaml"
        if env_ckpt:
            ckpt_path = Path(env_ckpt)
            if ckpt_path.exists():
                return env_cfg or default_config, ckpt_path, ckpt_path.name

        candidates = [
            self.workspace_root / "checkpoints/SAM2-Plus/checkpoint_phase123.pt",
            self.package_root / "checkpoints/SAM2-Plus/checkpoint_phase123.pt",
            self.package_root / "checkpoints/checkpoint_phase123.pt",
        ]
        for path in candidates:
            if path.exists():
                return default_config, path, path.name

        raise RuntimeError(
            "No SAM2-Plus checkpoint was found. Set ROBOCUP_SAM2_PLUS_CHECKPOINT or place checkpoint_phase123.pt in a SAM2-Plus checkpoints directory."
        )

    def _ensure_cotracker_loaded(self) -> None:
        if self.cotracker_predictor is not None:
            return

        try:
            import torch
            from cotracker.predictor import CoTrackerPredictor
        except Exception as exc:  # noqa: BLE001
            self.cotracker_last_error = str(exc)
            raise RuntimeError(
                "Could not import CoTracker. Install cotracker and make sure its dependencies are available."
            ) from exc

        checkpoint_path = self._discover_cotracker_checkpoint()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._configure_torch_runtime(torch, device)
        predictor = CoTrackerPredictor(checkpoint=str(checkpoint_path))
        if device == "cuda":
            predictor = predictor.cuda()
        predictor.eval()

        self.cotracker_predictor = predictor
        self.cotracker_model_name = checkpoint_path.name
        self.cotracker_device = device
        self.cotracker_last_error = None
        if self.torch is None:
            self.torch = torch

    def _discover_cotracker_checkpoint(self) -> Path:
        env_ckpt = os.environ.get("ROBOCUP_COTRACKER_CHECKPOINT")
        if env_ckpt:
            ckpt_path = Path(env_ckpt)
            if ckpt_path.exists():
                return ckpt_path

        candidates = [
            self.workspace_root / "robocuplabeling/co-tracker/checkpoints/scaled_offline.pth",
            self.workspace_root / "cotracker/checkpoints/scaled_offline.pth",
            self.package_root / "checkpoints/scaled_offline.pth",
        ]
        for path in candidates:
            if path.exists():
                return path

        raise RuntimeError(
            "No CoTracker checkpoint was found. Set ROBOCUP_COTRACKER_CHECKPOINT or place scaled_offline.pth in a checkpoints directory."
        )

    def _run_mask_propagation(
        self,
        folder_path: Path,
        frame_names: list[str],
        ordered_frames: list[TaskFrameSelection],
        completed_steps: int,
        total_steps: int,
        progress_callback=None,
    ) -> tuple[dict[int, list[MaskObject]], int]:
        object_meta: dict[str, tuple[int, str]] = {}
        for frame in ordered_frames:
            for obj in frame.objects:
                object_meta.setdefault(obj.object_key, (obj.object_id, obj.label))

        key_to_predictor_id = {
            object_key: index
            for index, object_key in enumerate(sorted(object_meta), start=1)
        }
        predictor_to_key = {
            predictor_id: object_key
            for object_key, predictor_id in key_to_predictor_id.items()
        }

        inference_state = self.mask_predictor.init_state(
            video_path=str(folder_path),
            frame_names=frame_names,
        )
        results: dict[int, list[MaskObject]] = {
            frame.frame_index: [] for frame in ordered_frames
        }
        try:
            with self.torch.inference_mode():
                with self._autocast_context():
                    for subset_frame_idx, frame in enumerate(ordered_frames):
                        for obj in frame.objects:
                            predictor_obj_id = key_to_predictor_id.get(obj.object_key)
                            if predictor_obj_id is None:
                                continue
                            self.mask_predictor.add_new_mask(
                                inference_state=inference_state,
                                frame_idx=subset_frame_idx,
                                obj_id=predictor_obj_id,
                                mask=obj.mask,
                            )

                    for (
                        subset_frame_idx,
                        out_obj_ids,
                        out_mask_logits,
                        _out_boxes_xyxy,
                        out_obj_score_logits,
                    ) in self.mask_predictor.propagate_in_video(inference_state):
                        original_frame_idx = ordered_frames[subset_frame_idx].frame_index
                        frame_results: list[MaskObject] = []
                        for obj_index, predictor_obj_id in enumerate(out_obj_ids):
                            object_key = predictor_to_key.get(int(predictor_obj_id))
                            if object_key is None:
                                continue
                            object_id, label = object_meta[object_key]
                            object_score = float(out_obj_score_logits[obj_index].squeeze().item())
                            mask_array = (out_mask_logits[obj_index] > self.score_thresh).cpu().numpy()
                            mask_array = np.asarray(mask_array).astype(bool).squeeze()
                            if object_score <= self.score_thresh or mask_array.ndim != 2 or not np.any(mask_array):
                                continue
                            bbox = self._mask_bbox(mask_array)
                            if bbox is None:
                                continue
                            frame_results.append(
                                MaskObject(
                                    object_id=object_id,
                                    label=label,
                                    mask=mask_array.copy(),
                                    bbox=bbox,
                                )
                            )
                        results[original_frame_idx] = frame_results
                        completed_steps += 1
                        if progress_callback is not None:
                            progress_callback(completed_steps, total_steps)
        finally:
            self.mask_predictor.reset_state(inference_state)

        return results, completed_steps

    def _run_point_propagation(
        self,
        folder_path: Path,
        frame_names: list[str],
        ordered_frames: list[TaskFrameSelection],
        completed_steps: int,
        total_steps: int,
        progress_callback=None,
    ) -> tuple[dict[int, list[KeypointAnnotation]], int]:
        point_meta: dict[str, int] = {}
        for frame in ordered_frames:
            for point in frame.points:
                point_meta.setdefault(point.point_key, point.point_id)

        key_to_predictor_id = {
            point_key: index
            for index, point_key in enumerate(sorted(point_meta), start=1)
        }
        predictor_to_key = {
            predictor_id: point_key
            for point_key, predictor_id in key_to_predictor_id.items()
        }

        inference_state = self.point_predictor.init_state(
            video_path=str(folder_path),
            frame_names=frame_names,
        )
        results: dict[int, list[KeypointAnnotation]] = {
            frame.frame_index: [] for frame in ordered_frames
        }
        try:
            with self.torch.inference_mode():
                with self._autocast_context():
                    for subset_frame_idx, frame in enumerate(ordered_frames):
                        for point in frame.points:
                            predictor_obj_id = key_to_predictor_id.get(point.point_key)
                            if predictor_obj_id is None:
                                continue
                            self.point_predictor.add_new_points_and_generate_gaussian_mask(
                                inference_state=inference_state,
                                frame_idx=subset_frame_idx,
                                obj_id=predictor_obj_id,
                                points=np.array([[point.x, point.y]], dtype=np.float32),
                                labels=np.array([1], dtype=np.int32),
                                radius=self.point_radius,
                                sigma=self.point_sigma,
                            )

                    for (
                        subset_frame_idx,
                        out_obj_ids,
                        out_mask_logits,
                        _out_boxes_xyxy,
                        out_obj_score_logits,
                    ) in self.point_predictor.propagate_in_video(inference_state):
                        original_frame_idx = ordered_frames[subset_frame_idx].frame_index
                        frame_results: list[KeypointAnnotation] = []
                        for obj_index, predictor_obj_id in enumerate(out_obj_ids):
                            point_key = predictor_to_key.get(int(predictor_obj_id))
                            if point_key is None:
                                continue
                            point_score = float(out_obj_score_logits[obj_index].squeeze().item())
                            if point_score <= self.score_thresh:
                                continue
                            logits = out_mask_logits[obj_index]
                            while logits.ndim > 2:
                                logits = logits.squeeze(0)
                            max_index = self.torch.argmax(logits)
                            max_score_y, max_score_x = self.torch.unravel_index(
                                max_index,
                                logits.shape,
                            )
                            frame_results.append(
                                KeypointAnnotation(
                                    point_id=point_meta[point_key],
                                    x=float(max_score_x.cpu().item()),
                                    y=float(max_score_y.cpu().item()),
                                )
                            )
                        results[original_frame_idx] = frame_results
                        completed_steps += 1
                        if progress_callback is not None:
                            progress_callback(completed_steps, total_steps)
        finally:
            self.point_predictor.reset_state(inference_state)

        return results, completed_steps

    def _run_cotracker_queries(
        self,
        frame_paths: list[Path],
        query_specs: list[tuple[int, float, float]],
        *,
        backward_tracking: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_cotracker_loaded()
        if self.cotracker_predictor is None or self.torch is None:
            raise RuntimeError("CoTracker failed to initialize.")

        if not query_specs:
            return (
                np.zeros((len(frame_paths), 0, 2), dtype=np.float32),
                np.zeros((len(frame_paths), 0), dtype=bool),
            )

        video_frames: list[np.ndarray] = []
        frame_size: tuple[int, int] | None = None
        for frame_path in frame_paths:
            with Image.open(frame_path) as image_handle:
                frame_array = np.array(image_handle.convert("RGB"))
            current_size = (frame_array.shape[0], frame_array.shape[1])
            if frame_size is None:
                frame_size = current_size
            elif current_size != frame_size:
                raise RuntimeError(
                    "CoTracker point propagation requires all selected frames to have the same resolution."
                )
            video_frames.append(frame_array)

        video_tensor = self.torch.from_numpy(np.stack(video_frames, axis=0))
        video_tensor = video_tensor.permute(0, 3, 1, 2)[None].float()
        if self.cotracker_device == "cuda":
            video_tensor = video_tensor.cuda()

        queries = self.torch.tensor(
            [[float(frame_idx), x, y] for frame_idx, x, y in query_specs],
            dtype=self.torch.float32,
        )[None]
        if self.cotracker_device == "cuda":
            queries = queries.cuda()

        with self.torch.inference_mode():
            pred_tracks, pred_visibility = self.cotracker_predictor(
                video_tensor,
                queries=queries,
                backward_tracking=backward_tracking,
            )

        pred_tracks = pred_tracks[0].detach().cpu().numpy()
        pred_visibility = pred_visibility[0].detach().cpu().numpy()
        while pred_visibility.ndim > 2:
            pred_visibility = np.squeeze(pred_visibility, axis=-1)
        visibility_mask = pred_visibility > 0.5
        return pred_tracks, visibility_mask

    def _run_cotracker_point_propagation(
        self,
        frame_paths: list[Path],
        ordered_frames: list[TaskFrameSelection],
        *,
        backward_tracking: bool,
        completed_steps: int,
        total_steps: int,
        progress_callback=None,
    ) -> tuple[dict[int, list[KeypointAnnotation]], int]:
        query_specs: list[tuple[str, int, int, float, float]] = []
        for subset_frame_idx, frame in enumerate(ordered_frames):
            for point in frame.points:
                query_specs.append(
                    (
                        point.point_key,
                        point.point_id,
                        subset_frame_idx,
                        float(point.x),
                        float(point.y),
                    )
                )

        results: dict[int, list[KeypointAnnotation]] = {
            frame.frame_index: [] for frame in ordered_frames
        }
        if not query_specs:
            return results, completed_steps

        pred_tracks, visibility_mask = self._run_cotracker_queries(
            frame_paths=frame_paths,
            query_specs=[
                (frame_idx, x, y)
                for _point_key, _point_id, frame_idx, x, y in query_specs
            ],
            backward_tracking=backward_tracking,
        )

        for subset_frame_idx, frame in enumerate(ordered_frames):
            nearest_point_predictions: dict[str, tuple[int, int, float, float]] = {}
            for query_index, (point_key, point_id, query_frame_idx, _x, _y) in enumerate(query_specs):
                if not backward_tracking and subset_frame_idx < query_frame_idx:
                    continue
                if not bool(visibility_mask[subset_frame_idx, query_index]):
                    continue
                xy = pred_tracks[subset_frame_idx, query_index]
                frame_distance = abs(subset_frame_idx - query_frame_idx)
                current_best = nearest_point_predictions.get(point_key)
                if current_best is not None and current_best[0] <= frame_distance:
                    continue
                nearest_point_predictions[point_key] = (
                    frame_distance,
                    point_id,
                    float(xy[0]),
                    float(xy[1]),
                )

            frame_points: list[KeypointAnnotation] = []
            for _point_key, (_distance, point_id, x_value, y_value) in nearest_point_predictions.items():
                frame_points.append(
                    KeypointAnnotation(
                        point_id=point_id,
                        x=x_value,
                        y=y_value,
                    )
                )
            results[frame.frame_index] = frame_points
            completed_steps += 1
            if progress_callback is not None:
                progress_callback(completed_steps, total_steps)

        return results, completed_steps

    def _merge_object_results(
        self,
        task: PropagationTask,
        forward_results: dict[int, list[MaskObject]],
        reverse_results: dict[int, list[MaskObject]],
    ) -> dict[int, list[MaskObject]]:
        pair_to_key: dict[tuple[int, str], str] = {}
        for frame in task.frames:
            for obj in frame.objects:
                pair_to_key.setdefault((obj.object_id, obj.label), obj.object_key)

        merged: dict[int, list[MaskObject]] = {}
        for frame in task.frames:
            frame_index = frame.frame_index
            forward_map = {
                pair_to_key.get((obj.object_id, obj.label), ""): obj
                for obj in forward_results.get(frame_index, [])
            }
            reverse_map = {
                pair_to_key.get((obj.object_id, obj.label), ""): obj
                for obj in reverse_results.get(frame_index, [])
            }
            frame_objects: list[MaskObject] = []
            for object_key in task.object_keys:
                obj = forward_map.get(object_key) or reverse_map.get(object_key)
                if obj is not None:
                    frame_objects.append(obj)
            merged[frame_index] = frame_objects
        return merged

    def _merge_point_results(
        self,
        task: PropagationTask,
        forward_results: dict[int, list[KeypointAnnotation]],
        reverse_results: dict[int, list[KeypointAnnotation]],
    ) -> dict[int, list[KeypointAnnotation]]:
        id_to_key: dict[int, str] = {}
        for frame in task.frames:
            for point in frame.points:
                id_to_key.setdefault(point.point_id, point.point_key)

        merged: dict[int, list[KeypointAnnotation]] = {}
        for frame in task.frames:
            frame_index = frame.frame_index
            forward_map = {
                id_to_key.get(point.point_id, ""): point
                for point in forward_results.get(frame_index, [])
            }
            reverse_map = {
                id_to_key.get(point.point_id, ""): point
                for point in reverse_results.get(frame_index, [])
            }
            frame_points: list[KeypointAnnotation] = []
            for point_key in task.point_keys:
                point = forward_map.get(point_key) or reverse_map.get(point_key)
                if point is not None:
                    frame_points.append(point)
            merged[frame_index] = frame_points
        return merged

    def _autocast_context(self):
        if self.device == "cuda" and self.torch is not None:
            return self.torch.autocast(device_type="cuda", dtype=self.torch.bfloat16)
        return nullcontext()

    def _configure_torch_runtime(self, torch, device: str) -> None:
        if device != "cuda":
            return
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def _mask_bbox(self, mask: np.ndarray) -> tuple[int, int, int, int] | None:
        positions = np.argwhere(mask)
        if positions.size == 0:
            return None
        y_min, x_min = positions.min(axis=0)
        y_max, x_max = positions.max(axis=0)
        return int(x_min), int(y_min), int(x_max), int(y_max)


class ImageCanvas(QWidget):
    keypoint_clicked = pyqtSignal(float, float)
    sam_prompt_point_clicked = pyqtSignal(float, float)
    sam_prompt_box_created = pyqtSignal(float, float, float, float)

    def __init__(self) -> None:
        super().__init__()
        self.setMouseTracking(True)
        self.setMinimumSize(480, 320)

        self.image: QImage | None = None
        self.image_rect = QRectF()
        self.tool_mode = "point"
        self.sam_prompt_mode = "point"
        self.show_point_ids = True
        self.keypoints: list[KeypointAnnotation] = []
        self.sam_prompt_points: list[SamPromptPoint] = []
        self.sam_prompt_box: SamPromptBox | None = None
        self.preview_mask: np.ndarray | None = None
        self.objects: list[MaskObject] = []
        self.drag_start: QPointF | None = None
        self.drag_current: QPointF | None = None

    def set_image(self, image: QImage | None) -> None:
        self.image = image
        self.update()

    def set_tool_state(self, tool_mode: str, sam_prompt_mode: str) -> None:
        self.tool_mode = tool_mode
        self.sam_prompt_mode = sam_prompt_mode
        if tool_mode in {"point", "sam2"}:
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        else:
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def set_keypoints(self, keypoints: list[KeypointAnnotation]) -> None:
        self.keypoints = keypoints
        self.update()

    def set_show_point_ids(self, show_ids: bool) -> None:
        self.show_point_ids = show_ids
        self.update()

    def set_sam_prompts(
        self,
        prompt_points: list[SamPromptPoint],
        prompt_box: SamPromptBox | None,
    ) -> None:
        self.sam_prompt_points = prompt_points
        self.sam_prompt_box = prompt_box
        self.update()

    def set_preview_mask(self, preview_mask: np.ndarray | None) -> None:
        self.preview_mask = preview_mask
        self.update()

    def set_objects(self, objects: list[MaskObject]) -> None:
        self.objects = objects
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0f141a"))

        if self.image is None:
            painter.setPen(QColor("#95a1af"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Open an image folder with Ctrl+U",
            )
            self.image_rect = QRectF()
            return

        self.image_rect = self._fit_image_rect()
        painter.drawImage(self.image_rect, self.image)
        self._draw_objects(painter)
        self._draw_preview_mask(painter)
        self._draw_sam_prompts(painter)
        self._draw_keypoints(painter)
        self._draw_drag_box(painter)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        super().mousePressEvent(event)
        if event.button() != Qt.MouseButton.LeftButton or self.image is None:
            return

        widget_pos = QPointF(event.position())
        if not self.image_rect.contains(widget_pos):
            return

        if self.tool_mode == "point":
            image_pos = self._widget_to_image(widget_pos)
            if image_pos is not None:
                self.keypoint_clicked.emit(image_pos.x(), image_pos.y())
            return

        if self.tool_mode == "sam2":
            if self.sam_prompt_mode == "point":
                image_pos = self._widget_to_image(widget_pos)
                if image_pos is not None:
                    self.sam_prompt_point_clicked.emit(
                        image_pos.x(), image_pos.y()
                    )
            elif self.sam_prompt_mode == "box":
                self.drag_start = widget_pos
                self.drag_current = widget_pos
                self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        super().mouseMoveEvent(event)
        if self.drag_start is None:
            return
        self.drag_current = self._clamp_to_image_rect(QPointF(event.position()))
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        super().mouseReleaseEvent(event)
        if (
            self.tool_mode != "sam2"
            or self.sam_prompt_mode != "box"
            or self.drag_start is None
            or self.image is None
        ):
            self.drag_start = None
            self.drag_current = None
            return

        drag_end = self._clamp_to_image_rect(QPointF(event.position()))
        start_image = self._widget_to_image(self.drag_start)
        end_image = self._widget_to_image(drag_end)
        self.drag_start = None
        self.drag_current = None
        self.update()

        if start_image is None or end_image is None:
            return
        if abs(start_image.x() - end_image.x()) < 4 or abs(start_image.y() - end_image.y()) < 4:
            return

        self.sam_prompt_box_created.emit(
            start_image.x(),
            start_image.y(),
            end_image.x(),
            end_image.y(),
        )

    def _fit_image_rect(self) -> QRectF:
        if self.image is None:
            return QRectF()

        available_width = max(self.width() - 24, 1)
        available_height = max(self.height() - 24, 1)
        image_ratio = self.image.width() / max(self.image.height(), 1)
        area_ratio = available_width / max(available_height, 1)

        if image_ratio >= area_ratio:
            width = available_width
            height = width / max(image_ratio, 1e-6)
        else:
            height = available_height
            width = height * image_ratio

        x = (self.width() - width) / 2
        y = (self.height() - height) / 2
        return QRectF(x, y, width, height)

    def _draw_objects(self, painter: QPainter) -> None:
        for obj in self.objects:
            color = OBJECT_COLORS[(obj.object_id - 1) % len(OBJECT_COLORS)]
            self._draw_mask_overlay(painter, obj.mask, color)
            self._draw_object_badge(painter, obj, color)

    def _draw_preview_mask(self, painter: QPainter) -> None:
        if self.preview_mask is None:
            return
        self._draw_mask_overlay(painter, self.preview_mask, QColor(190, 190, 190, 95))

    def _draw_mask_overlay(
        self,
        painter: QPainter,
        mask: np.ndarray,
        color: QColor,
    ) -> None:
        if self.image is None or self.image_rect.isNull():
            return
        if mask.ndim != 2:
            return

        rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        rgba[mask] = [color.red(), color.green(), color.blue(), color.alpha()]
        overlay = QImage(
            rgba.data,
            rgba.shape[1],
            rgba.shape[0],
            rgba.shape[1] * 4,
            QImage.Format.Format_RGBA8888,
        ).copy()
        painter.drawImage(self.image_rect, overlay)

    def _draw_object_badge(
        self,
        painter: QPainter,
        obj: MaskObject,
        color: QColor,
    ) -> None:
        anchor = self._image_to_widget(float(obj.bbox[0]), float(obj.bbox[1]))
        if anchor is None:
            return

        label_rect = QRectF(anchor.x() + 6, anchor.y() + 6, 132, 24)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(15, 20, 26, 220))
        painter.drawRoundedRect(label_rect, 7, 7)
        painter.setBrush(color)
        painter.drawEllipse(QPointF(label_rect.left() + 12, label_rect.center().y()), 5, 5)
        painter.setPen(QColor("#f5f7fa"))
        painter.drawText(
            QRectF(label_rect.left() + 22, label_rect.top(), label_rect.width() - 24, label_rect.height()),
            Qt.AlignmentFlag.AlignVCenter,
            f"#{obj.object_id} {obj.label}",
        )

    def _draw_sam_prompts(self, painter: QPainter) -> None:
        prompt_pen = QPen(QColor("#ffb347"))
        prompt_pen.setWidth(2)
        prompt_pen.setStyle(Qt.PenStyle.DashLine)
        point_outline = QPen(QColor("#fff4db"))
        point_outline.setWidth(2)

        if self.sam_prompt_box is not None:
            box = self.sam_prompt_box.normalized()
            top_left = self._image_to_widget(box.x1, box.y1)
            bottom_right = self._image_to_widget(box.x2, box.y2)
            if top_left is not None and bottom_right is not None:
                painter.setPen(prompt_pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(QRectF(top_left, bottom_right))

        for point in self.sam_prompt_points:
            widget_point = self._image_to_widget(point.x, point.y)
            if widget_point is None:
                continue
            painter.setPen(point_outline)
            painter.setBrush(QColor("#ffb347"))
            painter.drawEllipse(widget_point, 7, 7)
            painter.setPen(QPen(QColor("#ffb347"), 2))
            painter.drawLine(QLineF(widget_point.x() - 10, widget_point.y(), widget_point.x() + 10, widget_point.y()))
            painter.drawLine(QLineF(widget_point.x(), widget_point.y() - 10, widget_point.x(), widget_point.y() + 10))

    def _draw_keypoints(self, painter: QPainter) -> None:
        if self.image is None or self.image_rect.isNull():
            return

        point_pen = QPen(QColor("#2d72d9"))
        point_pen.setWidth(2)
        outline_pen = QPen(QColor("#f5f7fa"))
        outline_pen.setWidth(2)
        text_pen = QPen(QColor("#f5f7fa"))

        for point in self.keypoints:
            widget_point = self._image_to_widget(point.x, point.y)
            if widget_point is None:
                continue

            painter.setPen(outline_pen)
            painter.setBrush(QColor("#2d72d9"))
            painter.drawEllipse(widget_point, 6, 6)

            painter.setPen(point_pen)
            painter.drawLine(QLineF(widget_point.x() - 9, widget_point.y(), widget_point.x() + 9, widget_point.y()))
            painter.drawLine(QLineF(widget_point.x(), widget_point.y() - 9, widget_point.x(), widget_point.y() + 9))

            if self.show_point_ids:
                label_rect = QRectF(widget_point.x() + 9, widget_point.y() - 18, 56, 20)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(15, 20, 26, 220))
                painter.drawRoundedRect(label_rect, 6, 6)
                painter.setPen(text_pen)
                painter.drawText(
                    label_rect,
                    Qt.AlignmentFlag.AlignCenter,
                    str(point.point_id),
                )

    def _draw_drag_box(self, painter: QPainter) -> None:
        if self.drag_start is None or self.drag_current is None:
            return

        painter.setPen(QPen(QColor("#ffb347"), 2, Qt.PenStyle.DashLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(QRectF(self.drag_start, self.drag_current))

    def _clamp_to_image_rect(self, position: QPointF) -> QPointF:
        if self.image_rect.isNull():
            return position
        return QPointF(
            min(max(position.x(), self.image_rect.left()), self.image_rect.right()),
            min(max(position.y(), self.image_rect.top()), self.image_rect.bottom()),
        )

    def _widget_to_image(self, position: QPointF) -> QPointF | None:
        if self.image is None or self.image_rect.isNull():
            return None

        x_ratio = (position.x() - self.image_rect.left()) / self.image_rect.width()
        y_ratio = (position.y() - self.image_rect.top()) / self.image_rect.height()
        x = min(max(x_ratio, 0.0), 1.0) * self.image.width()
        y = min(max(y_ratio, 0.0), 1.0) * self.image.height()
        return QPointF(x, y)

    def _image_to_widget(self, x: float, y: float) -> QPointF | None:
        if self.image is None or self.image_rect.isNull():
            return None

        widget_x = self.image_rect.left() + (x / max(self.image.width(), 1)) * self.image_rect.width()
        widget_y = self.image_rect.top() + (y / max(self.image.height(), 1)) * self.image_rect.height()
        return QPointF(widget_x, widget_y)


class ObjectTimelineWidget(QWidget):
    selection_changed = pyqtSignal(int, int)
    object_toggled = pyqtSignal(str, bool)
    frame_double_clicked = pyqtSignal(int)

    row_height = 34
    header_height = 44
    label_width = 230
    cell_width = 20
    checkbox_size = 16

    def __init__(self) -> None:
        super().__init__()
        self.frame_paths: list[Path] = []
        self.rows: list[TimelineObjectRow] = []
        self.current_frame_index = -1
        self.selected_range: tuple[int, int] | None = None
        self.active_object_keys: set[str] = set()
        self.drag_anchor_frame: int | None = None
        self.row_header_title = "Objects"
        self.empty_message = "Finished objects will appear here once you create them."
        self.setMinimumSize(420, 180)

    def sizeHint(self) -> QSize:  # noqa: N802
        frame_count = len(self.frame_paths)
        row_count = max(len(self.rows), 1)
        width = self.label_width + max(frame_count, 1) * self.cell_width + 24
        height = self.header_height + row_count * self.row_height + 16
        return QSize(width, height)

    def set_timeline_data(
        self,
        frame_paths: list[Path],
        rows: list[TimelineObjectRow],
        current_frame_index: int,
        selected_range: tuple[int, int] | None,
        active_object_keys: set[str],
        row_header_title: str = "Objects",
        empty_message: str = "Finished objects will appear here once you create them.",
    ) -> None:
        self.frame_paths = frame_paths
        self.rows = rows
        self.current_frame_index = current_frame_index
        self.selected_range = selected_range
        self.active_object_keys = set(active_object_keys)
        self.row_header_title = row_header_title
        self.empty_message = empty_message
        size = self.sizeHint()
        self.resize(size)
        self.setMinimumSize(size)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0f141a"))

        header_rect = QRectF(0, 0, self.width(), self.header_height)
        painter.fillRect(header_rect, QColor("#10151c"))
        painter.setPen(QColor("#2a3442"))
        painter.drawLine(0, self.header_height - 1, self.width(), self.header_height - 1)
        painter.drawLine(self.label_width, 0, self.label_width, self.height())

        if not self.frame_paths:
            painter.setPen(QColor("#95a1af"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Load an image folder to build the timeline.",
            )
            return

        self._draw_selection_background(painter)
        self._draw_header(painter)

        if not self.rows:
            empty_rect = QRectF(
                self.label_width,
                self.header_height,
                max(self.width() - self.label_width, 0),
                max(self.height() - self.header_height, 0),
            )
            painter.setPen(QColor("#8793a1"))
            painter.drawText(
                empty_rect,
                Qt.AlignmentFlag.AlignCenter,
                self.empty_message,
            )
            return

        for row_index, row in enumerate(self.rows):
            self._draw_row(painter, row_index, row)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        super().mousePressEvent(event)
        if event.button() != Qt.MouseButton.LeftButton:
            return

        position = QPointF(event.position())
        row_index = self._row_index_at_y(position.y())
        if row_index is not None and self._checkbox_rect(row_index).contains(position):
            if self.selected_range is None:
                return
            object_key = self.rows[row_index].object_key
            is_active = object_key not in self.active_object_keys
            self.object_toggled.emit(object_key, is_active)
            return

        frame_index = self._frame_index_at_x(position.x())
        if frame_index is None:
            return

        self.drag_anchor_frame = frame_index
        self.selection_changed.emit(frame_index, frame_index)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        super().mouseMoveEvent(event)
        if self.drag_anchor_frame is None:
            return

        frame_index = self._frame_index_at_x(QPointF(event.position()).x())
        if frame_index is None:
            return
        self.selection_changed.emit(self.drag_anchor_frame, frame_index)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        super().mouseReleaseEvent(event)
        self.drag_anchor_frame = None

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        super().mouseDoubleClickEvent(event)
        if event.button() != Qt.MouseButton.LeftButton:
            return

        frame_index = self._frame_index_at_x(QPointF(event.position()).x())
        if frame_index is not None:
            self.frame_double_clicked.emit(frame_index)

    def _draw_selection_background(self, painter: QPainter) -> None:
        if self.selected_range is None:
            return

        start_index, end_index = self.selected_range
        left = self.label_width + start_index * self.cell_width
        width = (end_index - start_index + 1) * self.cell_width
        selection_rect = QRectF(left, 0, width, self.height())
        painter.fillRect(selection_rect, QColor(45, 114, 217, 42))

    def _draw_header(self, painter: QPainter) -> None:
        painter.setPen(QColor("#d8e1e8"))
        painter.drawText(
            QRectF(14, 0, self.label_width - 28, self.header_height),
            Qt.AlignmentFlag.AlignVCenter,
            self.row_header_title,
        )

        frame_count = len(self.frame_paths)
        for frame_index in range(frame_count):
            cell_rect = self._cell_rect(frame_index, -1)
            if frame_index == self.current_frame_index:
                painter.fillRect(cell_rect, QColor(255, 255, 255, 28))

            if (
                frame_count <= 14
                or frame_index in {0, self.current_frame_index, frame_count - 1}
                or frame_index % 5 == 0
            ):
                painter.setPen(QColor("#95a1af"))
                painter.drawText(
                    cell_rect,
                    Qt.AlignmentFlag.AlignCenter,
                    str(frame_index + 1),
                )

            painter.setPen(QColor("#22303d"))
            painter.drawLine(
                int(cell_rect.right()),
                self.header_height - 12,
                int(cell_rect.right()),
                self.height(),
            )

    def _draw_row(self, painter: QPainter, row_index: int, row: TimelineObjectRow) -> None:
        top = self.header_height + row_index * self.row_height
        row_rect = QRectF(0, top, self.width(), self.row_height)

        painter.fillRect(
            row_rect,
            QColor("#121820") if row_index % 2 == 0 else QColor("#10161e"),
        )
        painter.setPen(QColor("#1f2a36"))
        painter.drawLine(0, int(row_rect.bottom()), self.width(), int(row_rect.bottom()))

        checkbox_rect = self._checkbox_rect(row_index)
        checkbox_color = QColor("#2d72d9") if row.object_key in self.active_object_keys else QColor("#161b22")
        border_color = QColor("#2d72d9") if row.object_key in self.active_object_keys else QColor("#536170")
        if self.selected_range is None:
            checkbox_color = QColor("#121820")
            border_color = QColor("#394655")
        painter.setPen(QPen(border_color, 1.5))
        painter.setBrush(checkbox_color)
        painter.drawRoundedRect(checkbox_rect, 3, 3)

        if row.object_key in self.active_object_keys:
            painter.setPen(QPen(QColor("#f5f7fa"), 2))
            painter.drawLine(
                QLineF(
                    checkbox_rect.left() + 3,
                    checkbox_rect.center().y(),
                    checkbox_rect.center().x() - 1,
                    checkbox_rect.bottom() - 4,
                )
            )
            painter.drawLine(
                QLineF(
                    checkbox_rect.center().x() - 1,
                    checkbox_rect.bottom() - 4,
                    checkbox_rect.right() - 3,
                    checkbox_rect.top() + 4,
                )
            )

        painter.setPen(QColor("#d8e1e8"))
        painter.drawText(
            QRectF(40, top, self.label_width - 50, self.row_height),
            Qt.AlignmentFlag.AlignVCenter,
            row.object_key,
        )

        for frame_index, visible in enumerate(row.visibility):
            cell_rect = self._cell_rect(frame_index, row_index)
            if frame_index == self.current_frame_index:
                painter.fillRect(cell_rect, QColor(255, 255, 255, 18))

            if visible:
                fill_color = QColor("#6ea8fe") if row.object_key in self.active_object_keys else QColor("#6b7a8a")
                box_rect = QRectF(
                    cell_rect.left() + 4,
                    cell_rect.top() + 8,
                    max(cell_rect.width() - 8, 4),
                    max(cell_rect.height() - 16, 4),
                )
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(fill_color)
                painter.drawRoundedRect(box_rect, 3, 3)
            else:
                line_color = QColor("#415264") if row.object_key in self.active_object_keys else QColor("#313e4c")
                painter.setPen(QPen(line_color, 2))
                painter.drawLine(
                    QLineF(
                        cell_rect.left() + 3,
                        cell_rect.center().y(),
                        cell_rect.right() - 3,
                        cell_rect.center().y(),
                    )
                )

    def _row_index_at_y(self, y: float) -> int | None:
        if y < self.header_height:
            return None
        row_index = int((y - self.header_height) // self.row_height)
        if row_index < 0 or row_index >= len(self.rows):
            return None
        return row_index

    def _frame_index_at_x(self, x: float) -> int | None:
        if x < self.label_width:
            return None
        frame_index = int((x - self.label_width) // self.cell_width)
        if frame_index < 0 or frame_index >= len(self.frame_paths):
            return None
        return frame_index

    def _checkbox_rect(self, row_index: int) -> QRectF:
        top = self.header_height + row_index * self.row_height
        return QRectF(
            14,
            top + (self.row_height - self.checkbox_size) / 2,
            self.checkbox_size,
            self.checkbox_size,
        )

    def _cell_rect(self, frame_index: int, row_index: int) -> QRectF:
        top = 0 if row_index < 0 else self.header_height + row_index * self.row_height
        height = self.header_height if row_index < 0 else self.row_height
        left = self.label_width + frame_index * self.cell_width
        return QRectF(left, top, self.cell_width, height)


class TaskListItemWidget(QWidget):
    start_requested = pyqtSignal(int)

    def __init__(self, row_index: int, task: PropagationTask) -> None:
        super().__init__()
        self.row_index = row_index

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        self.title_label = QLabel()
        self.title_label.setObjectName("panelPrimary")
        top_row.addWidget(self.title_label, 1)

        self.status_label = QLabel()
        self.status_label.setObjectName("panelMuted")
        top_row.addWidget(self.status_label)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(
            lambda: self.start_requested.emit(self.row_index)
        )
        top_row.addWidget(self.start_button)
        layout.addLayout(top_row)

        self.summary_label = QLabel()
        self.summary_label.setObjectName("panelMuted")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        self.update_task(task, row_index)

    def update_task(self, task: PropagationTask, row_index: int) -> None:
        self.row_index = row_index
        frame_total = task.end_index - task.start_index + 1
        annotated_frames = sum(
            1 for frame in task.frames if frame.objects or frame.points
        )
        point_summary = (
            f"{len(task.point_keys)} points via {_point_tracker_display_name(task.point_tracker)}"
            if task.point_keys
            else "0 points"
        )
        self.title_label.setText(
            f"Task {task.task_id}  Frames {task.start_index + 1}-{task.end_index + 1}"
        )
        self.status_label.setText(task.status)
        self.summary_label.setText(
            (
                f"{len(task.object_keys)} objects, {point_summary}, "
                f"{annotated_frames}/{frame_total} frames seeded, "
                f"{'reverse fill on' if task.use_reverse_pass else 'forward only'}."
            )
        )
        self.progress_bar.setValue(max(0, min(task.progress, 100)))
        is_running = task.status == "Running"
        self.progress_bar.setVisible(is_running)
        self.start_button.setEnabled(not is_running)
        self.start_button.setText("Running..." if is_running else "Start")


class SceneTaskListItemWidget(QWidget):
    load_requested = pyqtSignal(int)

    def __init__(self, row_index: int, task: SceneBackgroundTask) -> None:
        super().__init__()
        self.row_index = row_index

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        self.title_label = QLabel()
        self.title_label.setObjectName("panelPrimary")
        top_row.addWidget(self.title_label, 1)

        self.status_label = QLabel()
        self.status_label.setObjectName("panelMuted")
        top_row.addWidget(self.status_label)

        self.load_button = QPushButton("Load")
        self.load_button.clicked.connect(
            lambda: self.load_requested.emit(self.row_index)
        )
        top_row.addWidget(self.load_button)
        layout.addLayout(top_row)

        self.summary_label = QLabel()
        self.summary_label.setObjectName("panelMuted")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        self.update_task(task, row_index)

    def update_task(self, task: SceneBackgroundTask, row_index: int) -> None:
        self.row_index = row_index
        self.title_label.setText(
            f"Scene {task.task_id}  {task.request.scene_name}"
        )
        self.status_label.setText(task.status)
        self.summary_label.setText(task.message or "Waiting to start")
        self.progress_bar.setValue(max(0, min(task.progress, 100)))
        self.progress_bar.setVisible(task.status in {"Queued", "Running"})
        self.load_button.setEnabled(task.status in {"Completed", "Failed"})


class LabelingWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RoboCup Labeling")
        self.resize(1520, 940)
        self.setMinimumSize(1180, 760)

        self.app_root = Path(__file__).resolve().parent.parent
        self.workspace_root = self.app_root.parent
        self.sam2_bundle_root = self.app_root / "SAM2-Plus"

        self.folder_path: Path | None = None
        self.image_paths: list[Path] = []
        self.current_index = -1
        self.current_image: QImage | None = None
        self.current_path: Path | None = None

        self.point_annotations: dict[str, list[KeypointAnnotation]] = {}
        self.sam2_prompt_points: dict[str, list[SamPromptPoint]] = {}
        self.sam2_prompt_boxes: dict[str, SamPromptBox | None] = {}
        self.sam2_preview_masks: dict[str, np.ndarray | None] = {}
        self.object_annotations: dict[str, list[MaskObject]] = {}
        self.propagation_tasks: list[PropagationTask] = []

        self.show_point_ids = True
        self.active_tool = "point"
        self.sam2_prompt_mode = "point"
        self.last_object_label = "object"
        self.timeline_panel_mode = "objects"
        self.selected_object_time_window: tuple[int, int] | None = None
        self.selected_point_time_window: tuple[int, int] | None = None
        self.selected_timeline_objects: set[str] = set()
        self.selected_timeline_points: set[str] = set()
        self.undo_stack: list[AppStateSnapshot] = []
        self.redo_stack: list[AppStateSnapshot] = []
        self.max_history_depth = 50
        self.scene_watch_root: Path | None = None
        self.scene_watch_poll_interval_ms = 3000
        self.scene_watch_processed: set[str] = set()
        self.scene_watch_requests: list[OpenVocabularyDetectionRequest] = []
        self.scene_background_tasks: list[SceneBackgroundTask] = []
        self.scene_task_update_queue: Queue[SceneTaskUpdate] = Queue()
        self.scene_task_worker_thread: threading.Thread | None = None
        self.active_scene_task_id: int | None = None
        self.scene_watch_detector = OpenVocabularyDetectorInterface()
        self.sam2_engine = Sam2ImageEngine(
            workspace_root=self.workspace_root,
            package_root=self.sam2_bundle_root,
        )
        self.sam2_video_engine = Sam2VideoPropagationEngine(
            workspace_root=self.workspace_root,
            package_root=self.sam2_bundle_root,
        )
        self.sam2_error_message: str | None = None
        self.scene_watch_timer = QTimer(self)
        self.scene_watch_timer.setInterval(self.scene_watch_poll_interval_ms)
        self.scene_watch_timer.timeout.connect(self._poll_incoming_scene_folders)
        self.scene_task_poll_timer = QTimer(self)
        self.scene_task_poll_timer.setInterval(150)
        self.scene_task_poll_timer.timeout.connect(self._drain_scene_task_updates)

        self._build_actions()
        self._build_ui()
        self._build_menus()
        self._apply_styles()
        self._set_sam2_prompt_mode("point")
        self._set_active_tool("point")
        self._show_empty_state()
        self._apply_default_scene_watch_root()
        self._update_scene_watch_widgets()
        self.scene_task_poll_timer.start()

    def _build_actions(self) -> None:
        self.open_folder_action = QAction("Open Image Folder", self)
        self.open_folder_action.setShortcut(QKeySequence("Ctrl+U"))
        self.open_folder_action.triggered.connect(self.open_folder_dialog)

        self.import_coco_action = QAction("Import COCO Annotations", self)
        self.import_coco_action.triggered.connect(self.import_coco_annotations)

        self.set_scene_watch_root_action = QAction("Set Incoming Scene Path", self)
        self.set_scene_watch_root_action.triggered.connect(
            self.choose_scene_watch_root
        )

        self.toggle_scene_watch_action = QAction("Watch Incoming Scenes", self)
        self.toggle_scene_watch_action.setCheckable(True)
        self.toggle_scene_watch_action.triggered.connect(
            self._toggle_scene_watch_from_action
        )

        self.scan_scene_watch_now_action = QAction("Scan Incoming Scenes Now", self)
        self.scan_scene_watch_now_action.triggered.connect(
            self.scan_incoming_scene_root_now
        )

        self.export_coco_action = QAction("Export COCO Annotations", self)
        self.export_coco_action.triggered.connect(
            lambda: self.export_coco_annotations(
                include_objects=True,
                include_points=True,
            )
        )

        self.export_coco_objects_only_action = QAction(
            "Export COCO Objects Only",
            self,
        )
        self.export_coco_objects_only_action.triggered.connect(
            lambda: self.export_coco_annotations(
                include_objects=True,
                include_points=False,
            )
        )

        self.export_coco_keypoints_only_action = QAction(
            "Export COCO Keypoints Only",
            self,
        )
        self.export_coco_keypoints_only_action.triggered.connect(
            lambda: self.export_coco_annotations(
                include_objects=False,
                include_points=True,
            )
        )

        self.finish_current_mask_action = QAction("Finish Current Mask", self)
        self.finish_current_mask_action.setShortcut(QKeySequence("F"))
        self.finish_current_mask_action.triggered.connect(self.finish_sam2_object)

        self.reset_sam2_selection_action = QAction("Reset SAM2 Selections", self)
        self.reset_sam2_selection_action.setShortcut(QKeySequence("B"))
        self.reset_sam2_selection_action.triggered.connect(
            self.clear_current_sam2_prompts
        )

        self.previous_action = QAction("Previous Image", self)
        self.previous_action.setShortcut(QKeySequence(Qt.Key.Key_Left))
        self.previous_action.triggered.connect(self.show_previous_image)

        self.next_action = QAction("Next Image", self)
        self.next_action.setShortcut(QKeySequence(Qt.Key.Key_Right))
        self.next_action.triggered.connect(self.show_next_image)

        self.undo_action = QAction("Undo", self)
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.undo_action.triggered.connect(self.undo_last_change)

        self.redo_action = QAction("Redo", self)
        self.redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        self.redo_action.triggered.connect(self.redo_last_change)

        self.show_ids_action = QAction("Show Point IDs", self)
        self.show_ids_action.setCheckable(True)
        self.show_ids_action.setChecked(True)
        self.show_ids_action.triggered.connect(self._toggle_point_ids)

        self.browse_tool_action = QAction("Browse Tool", self)
        self.browse_tool_action.setCheckable(True)
        self.browse_tool_action.triggered.connect(
            lambda _checked: self._set_active_tool("browse")
        )

        self.point_tool_action = QAction("Point Tool", self)
        self.point_tool_action.setCheckable(True)
        self.point_tool_action.triggered.connect(
            lambda _checked: self._set_active_tool("point")
        )

        self.sam2_tool_action = QAction("SAM2 Tool", self)
        self.sam2_tool_action.setCheckable(True)
        self.sam2_tool_action.triggered.connect(
            lambda _checked: self._set_active_tool("sam2")
        )

        self.sam2_point_prompt_action = QAction("SAM2 Point Prompt", self)
        self.sam2_point_prompt_action.setCheckable(True)
        self.sam2_point_prompt_action.triggered.connect(
            lambda _checked: self._set_sam2_prompt_mode("point")
        )

        self.sam2_box_prompt_action = QAction("SAM2 Box Prompt", self)
        self.sam2_box_prompt_action.setCheckable(True)
        self.sam2_box_prompt_action.triggered.connect(
            lambda _checked: self._set_sam2_prompt_mode("box")
        )

        self.create_task_action = QAction("Create Propagation Task", self)
        self.create_task_action.triggered.connect(
            self.create_propagation_task_from_selection
        )

        self.delete_timeline_objects_action = QAction(
            "Delete Selected Objects In Window",
            self,
        )
        self.delete_timeline_objects_action.triggered.connect(
            self.delete_selected_objects_in_window
        )

        self.clear_timeline_selection_action = QAction(
            "Clear Timeline Selection",
            self,
        )
        self.clear_timeline_selection_action.triggered.connect(
            self.clear_timeline_selection
        )

        self.select_all_timeline_frames_action = QAction(
            "Select All Timeline Frames",
            self,
        )
        self.select_all_timeline_frames_action.setShortcut(QKeySequence("Ctrl+A"))
        self.select_all_timeline_frames_action.setShortcutContext(
            Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        self.select_all_timeline_frames_action.triggered.connect(
            self.select_all_timeline_frames
        )

        self.tool_group = QActionGroup(self)
        self.tool_group.setExclusive(True)
        self.tool_group.addAction(self.browse_tool_action)
        self.tool_group.addAction(self.point_tool_action)
        self.tool_group.addAction(self.sam2_tool_action)

        self.sam2_prompt_group = QActionGroup(self)
        self.sam2_prompt_group.setExclusive(True)
        self.sam2_prompt_group.addAction(self.sam2_point_prompt_action)
        self.sam2_prompt_group.addAction(self.sam2_box_prompt_action)

        self.addAction(self.open_folder_action)
        self.addAction(self.undo_action)
        self.addAction(self.redo_action)
        self.addAction(self.previous_action)
        self.addAction(self.next_action)
        self.addAction(self.finish_current_mask_action)
        self.addAction(self.reset_sam2_selection_action)
        self._update_history_actions()

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self.open_folder_action)
        file_menu.addAction(self.set_scene_watch_root_action)
        file_menu.addAction(self.toggle_scene_watch_action)
        file_menu.addAction(self.scan_scene_watch_now_action)
        file_menu.addSeparator()
        file_menu.addAction(self.import_coco_action)
        file_menu.addSeparator()
        file_menu.addAction(self.previous_action)
        file_menu.addAction(self.next_action)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        edit_menu = self.menuBar().addMenu("Edit")
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)

        tools_menu = self.menuBar().addMenu("Tools")
        tools_menu.addAction(self.browse_tool_action)
        tools_menu.addAction(self.point_tool_action)
        tools_menu.addAction(self.sam2_tool_action)
        tools_menu.addSeparator()
        tools_menu.addAction(self.sam2_point_prompt_action)
        tools_menu.addAction(self.sam2_box_prompt_action)
        tools_menu.addSeparator()
        tools_menu.addAction(
            self.finish_current_mask_action,
        )
        tools_menu.addAction(
            self.reset_sam2_selection_action,
        )
        tools_menu.addSeparator()
        tools_menu.addAction(
            "Clear Points In Current Image",
            self.clear_current_image_points,
        )
        tools_menu.addSeparator()
        tools_menu.addAction(self.create_task_action)
        tools_menu.addAction(self.delete_timeline_objects_action)
        tools_menu.addSeparator()
        tools_menu.addAction(self.set_scene_watch_root_action)
        tools_menu.addAction(self.toggle_scene_watch_action)
        tools_menu.addAction(self.scan_scene_watch_now_action)

        export_menu = self.menuBar().addMenu("Export")
        export_menu.addAction(self.export_coco_action)
        export_menu.addAction(self.export_coco_objects_only_action)
        export_menu.addAction(self.export_coco_keypoints_only_action)

        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction("Fit To Window", lambda: self.canvas.update())
        view_menu.addSeparator()
        view_menu.addAction(self.show_ids_action)
        view_menu.addSeparator()
        toggle_panel_action = self.bottom_dock.toggleViewAction()
        toggle_panel_action.setText("Show Timeline / Tasks Panel")
        view_menu.addAction(toggle_panel_action)

        help_menu = self.menuBar().addMenu("Help")
        help_menu.addAction("About", self._show_about)

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self._build_header())

        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(10)
        content_layout.addWidget(self._build_tool_rail())
        content_layout.addWidget(self._build_workspace(), 1)
        content_layout.addWidget(self._build_sidebar())
        root_layout.addLayout(content_layout, 1)

        self.status_bar = QStatusBar()
        self.status_bar.showMessage("Ready")
        self.setStatusBar(self.status_bar)
        self.setCentralWidget(root)
        self._build_bottom_dock()

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("headerBar")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 14, 18, 14)
        header_layout.setSpacing(12)

        title_block = QVBoxLayout()
        title = QLabel("RoboCup Labeling")
        title.setObjectName("appTitle")
        subtitle = QLabel(
            "X-AnyLabeling-style workspace with keypoints, SAM2 prompts, and propagation task selection"
        )
        subtitle.setObjectName("appSubtitle")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        header_layout.addLayout(title_block)

        self.folder_summary = QLabel("No folder loaded")
        self.folder_summary.setObjectName("folderSummary")
        self.folder_summary.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        header_layout.addWidget(self.folder_summary, 1)

        self.open_button = QPushButton("Open Folder")
        self.open_button.setObjectName("accentButton")
        self.open_button.clicked.connect(self.open_folder_dialog)
        header_layout.addWidget(self.open_button)

        self.header_prev_button = QPushButton("◀")
        self.header_prev_button.clicked.connect(self.show_previous_image)
        header_layout.addWidget(self.header_prev_button)

        self.header_next_button = QPushButton("▶")
        self.header_next_button.clicked.connect(self.show_next_image)
        header_layout.addWidget(self.header_next_button)

        return header

    def _build_tool_rail(self) -> QWidget:
        rail = QFrame()
        rail.setObjectName("toolRail")
        rail.setFixedWidth(96)

        layout = QVBoxLayout(rail)
        layout.setContentsMargins(12, 16, 12, 16)
        layout.setSpacing(8)

        title = QLabel("TOOLS")
        title.setObjectName("railTitle")
        layout.addWidget(title)

        self.open_tool_button = QPushButton("Open")
        self.open_tool_button.setObjectName("railButton")
        self.open_tool_button.clicked.connect(self.open_folder_dialog)
        layout.addWidget(self.open_tool_button)

        self.prev_tool_button = QPushButton("Prev")
        self.prev_tool_button.setObjectName("railButton")
        self.prev_tool_button.clicked.connect(self.show_previous_image)
        layout.addWidget(self.prev_tool_button)

        self.next_tool_button = QPushButton("Next")
        self.next_tool_button.setObjectName("railButton")
        self.next_tool_button.clicked.connect(self.show_next_image)
        layout.addWidget(self.next_tool_button)

        self.browse_tool_button = QPushButton("Browse")
        self.browse_tool_button.setObjectName("toolToggleButton")
        self.browse_tool_button.setCheckable(True)
        self.browse_tool_button.clicked.connect(
            lambda _checked: self._set_active_tool("browse")
        )
        layout.addWidget(self.browse_tool_button)

        self.point_tool_button = QPushButton("Point")
        self.point_tool_button.setObjectName("toolToggleButton")
        self.point_tool_button.setCheckable(True)
        self.point_tool_button.clicked.connect(
            lambda _checked: self._set_active_tool("point")
        )
        layout.addWidget(self.point_tool_button)

        self.sam2_tool_button = QPushButton("SAM2")
        self.sam2_tool_button.setObjectName("toolToggleButton")
        self.sam2_tool_button.setCheckable(True)
        self.sam2_tool_button.clicked.connect(
            lambda _checked: self._set_active_tool("sam2")
        )
        layout.addWidget(self.sam2_tool_button)

        self.toggle_timeline_button = QPushButton("Panel")
        self.toggle_timeline_button.setObjectName("railButton")
        layout.addWidget(self.toggle_timeline_button)

        layout.addStretch(1)
        return rail

    def _build_workspace(self) -> QWidget:
        workspace = QWidget()
        layout = QVBoxLayout(workspace)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        nav_bar = QFrame()
        nav_bar.setObjectName("navBar")
        nav_layout = QHBoxLayout(nav_bar)
        nav_layout.setContentsMargins(12, 8, 12, 8)

        prev_button = QPushButton("◀ Previous")
        prev_button.clicked.connect(self.show_previous_image)
        nav_layout.addWidget(prev_button)

        self.image_counter = QLabel("0 / 0")
        self.image_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_counter.setObjectName("counterLabel")
        nav_layout.addWidget(self.image_counter, 1)

        next_button = QPushButton("Next ▶")
        next_button.clicked.connect(self.show_next_image)
        nav_layout.addWidget(next_button)

        layout.addWidget(nav_bar)

        self.canvas_frame = QFrame()
        self.canvas_frame.setObjectName("canvasFrame")
        canvas_layout = QVBoxLayout(self.canvas_frame)
        canvas_layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = ImageCanvas()
        self.canvas.keypoint_clicked.connect(self.add_keypoint)
        self.canvas.sam_prompt_point_clicked.connect(self.add_sam2_prompt_point)
        self.canvas.sam_prompt_box_created.connect(self.set_sam2_prompt_box)
        canvas_layout.addWidget(self.canvas)
        layout.addWidget(self.canvas_frame, 1)

        return workspace

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setFixedWidth(340)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        session_panel = self._build_panel("Session")
        session_layout = session_panel.layout()
        self.current_name_label = QLabel("Current image: -")
        self.current_name_label.setObjectName("panelPrimary")
        self.current_size_label = QLabel("Dimensions: -")
        self.current_size_label.setObjectName("panelMuted")
        self.folder_name_label = QLabel("Folder: -")
        self.folder_name_label.setObjectName("panelMuted")
        session_layout.addWidget(self.current_name_label)
        session_layout.addWidget(self.current_size_label)
        session_layout.addWidget(self.folder_name_label)

        self.scene_watch_status_label = QLabel("Incoming scene watch: off")
        self.scene_watch_status_label.setObjectName("panelPrimary")
        session_layout.addWidget(self.scene_watch_status_label)

        self.scene_watch_path_label = QLabel("Watch path: -")
        self.scene_watch_path_label.setObjectName("panelMuted")
        self.scene_watch_path_label.setWordWrap(True)
        session_layout.addWidget(self.scene_watch_path_label)

        self.scene_watch_result_label = QLabel("Detector hook: idle")
        self.scene_watch_result_label.setObjectName("panelMuted")
        self.scene_watch_result_label.setWordWrap(True)
        session_layout.addWidget(self.scene_watch_result_label)

        watch_button_row = QHBoxLayout()
        self.choose_watch_root_button = QPushButton("Watch Path")
        self.choose_watch_root_button.clicked.connect(self.choose_scene_watch_root)
        watch_button_row.addWidget(self.choose_watch_root_button)

        self.toggle_watch_button = QPushButton("Start Watch")
        self.toggle_watch_button.clicked.connect(self.toggle_scene_watch)
        watch_button_row.addWidget(self.toggle_watch_button)

        self.scan_watch_now_button = QPushButton("Scan Now")
        self.scan_watch_now_button.clicked.connect(self.scan_incoming_scene_root_now)
        watch_button_row.addWidget(self.scan_watch_now_button)
        session_layout.addLayout(watch_button_row)

        self.scene_watch_list = QListWidget()
        self.scene_watch_list.setMaximumHeight(120)
        session_layout.addWidget(self.scene_watch_list)
        layout.addWidget(session_panel)

        keypoint_panel = self._build_panel("Keypoints")
        keypoint_layout = keypoint_panel.layout()
        self.active_tool_label = QLabel("Active tool: Point")
        self.active_tool_label.setObjectName("panelPrimary")
        keypoint_layout.addWidget(self.active_tool_label)

        self.show_ids_checkbox = QCheckBox("Show point ids")
        self.show_ids_checkbox.setChecked(True)
        self.show_ids_checkbox.stateChanged.connect(
            lambda state: self._toggle_point_ids(state != 0)
        )
        keypoint_layout.addWidget(self.show_ids_checkbox)

        self.point_count_label = QLabel("Points in image: 0")
        self.point_count_label.setObjectName("panelMuted")
        keypoint_layout.addWidget(self.point_count_label)

        clear_points_button = QPushButton("Clear Current Image Points")
        clear_points_button.clicked.connect(self.clear_current_image_points)
        keypoint_layout.addWidget(clear_points_button)

        self.keypoint_list = QListWidget()
        keypoint_layout.addWidget(self.keypoint_list)
        layout.addWidget(keypoint_panel)

        sam2_panel = self._build_panel("SAM2")
        sam2_layout = sam2_panel.layout()
        self.sam2_mode_label = QLabel("Prompt mode: Point")
        self.sam2_mode_label.setObjectName("panelPrimary")
        sam2_layout.addWidget(self.sam2_mode_label)

        self.sam2_model_label = QLabel("Model: waiting")
        self.sam2_model_label.setObjectName("panelMuted")
        sam2_layout.addWidget(self.sam2_model_label)

        prompt_button_row = QHBoxLayout()
        self.sam2_point_prompt_button = QPushButton("Point Prompt")
        self.sam2_point_prompt_button.setObjectName("toolToggleButton")
        self.sam2_point_prompt_button.setCheckable(True)
        self.sam2_point_prompt_button.clicked.connect(
            lambda _checked: self._set_sam2_prompt_mode("point")
        )
        prompt_button_row.addWidget(self.sam2_point_prompt_button)

        self.sam2_box_prompt_button = QPushButton("Box Prompt")
        self.sam2_box_prompt_button.setObjectName("toolToggleButton")
        self.sam2_box_prompt_button.setCheckable(True)
        self.sam2_box_prompt_button.clicked.connect(
            lambda _checked: self._set_sam2_prompt_mode("box")
        )
        prompt_button_row.addWidget(self.sam2_box_prompt_button)
        sam2_layout.addLayout(prompt_button_row)

        self.sam2_prompt_count_label = QLabel("Prompts: 0 points, 0 boxes")
        self.sam2_prompt_count_label.setObjectName("panelMuted")
        sam2_layout.addWidget(self.sam2_prompt_count_label)

        self.sam2_preview_label = QLabel("Preview: none")
        self.sam2_preview_label.setObjectName("panelMuted")
        sam2_layout.addWidget(self.sam2_preview_label)

        finish_button = QPushButton("Finish Current Mask")
        finish_button.clicked.connect(self.finish_sam2_object)
        sam2_layout.addWidget(finish_button)

        clear_prompts_button = QPushButton("Clear SAM2 Prompts")
        clear_prompts_button.clicked.connect(self.clear_current_sam2_prompts)
        sam2_layout.addWidget(clear_prompts_button)

        self.sam2_prompt_list = QListWidget()
        sam2_layout.addWidget(self.sam2_prompt_list)
        layout.addWidget(sam2_panel)

        object_panel = self._build_panel("Objects")
        object_layout = object_panel.layout()
        self.object_count_label = QLabel("Objects in image: 0")
        self.object_count_label.setObjectName("panelPrimary")
        object_layout.addWidget(self.object_count_label)

        self.object_list = QListWidget()
        self.object_list.itemDoubleClicked.connect(self._edit_object_label_from_item)
        object_layout.addWidget(self.object_list)
        layout.addWidget(object_panel)

        files_panel = self._build_panel("Files")
        files_layout = files_panel.layout()
        self.file_list = QListWidget()
        self.file_list.itemSelectionChanged.connect(self._on_file_selected)
        files_layout.addWidget(self.file_list)
        layout.addWidget(files_panel, 1)

        return sidebar

    def _build_bottom_dock(self) -> None:
        self.bottom_dock = QDockWidget("Timeline / Tasks", self)
        self.bottom_dock.setObjectName("bottomDock")
        self.bottom_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self.bottom_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        dock_root = QWidget()
        dock_layout = QVBoxLayout(dock_root)
        dock_layout.setContentsMargins(10, 10, 10, 10)
        dock_layout.setSpacing(8)

        self.bottom_tabs = QTabWidget()
        self.timeline_tab = self._build_timeline_tab()
        self.tasks_tab = self._build_tasks_tab()
        self.bottom_tabs.addTab(self.timeline_tab, "Timeline")
        self.bottom_tabs.addTab(self.tasks_tab, "Tasks")
        dock_layout.addWidget(self.bottom_tabs)

        self.bottom_dock.setWidget(dock_root)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.bottom_dock)
        self.resizeDocks([self.bottom_dock], [270], Qt.Orientation.Vertical)
        self.toggle_timeline_button.clicked.connect(self._toggle_bottom_panel)

    def _build_timeline_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        info_card = QFrame()
        info_card.setObjectName("timelineInfoCard")
        info_layout = QVBoxLayout(info_card)
        info_layout.setContentsMargins(14, 12, 14, 12)
        info_layout.setSpacing(6)

        self.timeline_selection_label = QLabel(
            "Select a frame window directly on the timeline."
        )
        self.timeline_selection_label.setObjectName("panelPrimary")
        info_layout.addWidget(self.timeline_selection_label)

        self.timeline_objects_label = QLabel(
            "Objects can be activated per selected frame range."
        )
        self.timeline_objects_label.setObjectName("panelMuted")
        info_layout.addWidget(self.timeline_objects_label)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)
        self.object_timeline_button = QPushButton("Objects")
        self.object_timeline_button.setObjectName("toolToggleButton")
        self.object_timeline_button.setCheckable(True)
        self.object_timeline_button.clicked.connect(
            lambda _checked: self._set_timeline_panel_mode("objects")
        )
        mode_row.addWidget(self.object_timeline_button)

        self.point_timeline_button = QPushButton("Points")
        self.point_timeline_button.setObjectName("toolToggleButton")
        self.point_timeline_button.setCheckable(True)
        self.point_timeline_button.clicked.connect(
            lambda _checked: self._set_timeline_panel_mode("points")
        )
        mode_row.addWidget(self.point_timeline_button)
        mode_row.addStretch(1)
        info_layout.addLayout(mode_row)

        button_row = QHBoxLayout()
        button_row.setSpacing(8)

        self.create_task_button = QPushButton("Create Propagation Task")
        self.create_task_button.clicked.connect(
            self.create_propagation_task_from_selection
        )
        button_row.addWidget(self.create_task_button)

        self.reverse_task_checkbox = QCheckBox("Reverse Fill")
        self.reverse_task_checkbox.setChecked(False)
        button_row.addWidget(self.reverse_task_checkbox)

        self.point_tracker_selector = QComboBox()
        self.point_tracker_selector.addItem("Points: SAM2", "sam2")
        self.point_tracker_selector.addItem("Points: CoTracker", "cotracker")
        button_row.addWidget(self.point_tracker_selector)

        self.delete_timeline_objects_button = QPushButton(
            "Delete Objects In Window"
        )
        self.delete_timeline_objects_button.clicked.connect(
            self.delete_selected_objects_in_window
        )
        button_row.addWidget(self.delete_timeline_objects_button)

        self.clear_timeline_button = QPushButton("Clear Selection")
        self.clear_timeline_button.clicked.connect(self.clear_timeline_selection)
        button_row.addWidget(self.clear_timeline_button)
        button_row.addStretch(1)

        info_layout.addLayout(button_row)
        layout.addWidget(info_card)

        self.timeline_scroll = QScrollArea()
        self.timeline_scroll.setWidgetResizable(False)
        self.timeline_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.timeline_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.timeline_widget = ObjectTimelineWidget()
        self.timeline_widget.selection_changed.connect(
            self._on_timeline_selection_changed
        )
        self.timeline_widget.object_toggled.connect(self._on_timeline_row_toggled)
        self.timeline_widget.frame_double_clicked.connect(self.show_image_at_index)
        self.timeline_scroll.setWidget(self.timeline_widget)
        panel.addAction(self.select_all_timeline_frames_action)
        layout.addWidget(self.timeline_scroll, 1)

        return panel

    def _build_tasks_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        info_card = QFrame()
        info_card.setObjectName("timelineInfoCard")
        info_layout = QVBoxLayout(info_card)
        info_layout.setContentsMargins(14, 12, 14, 12)
        info_layout.setSpacing(6)

        self.task_summary_label = QLabel(
            "Tasks created from timeline selections will appear here."
        )
        self.task_summary_label.setObjectName("panelPrimary")
        info_layout.addWidget(self.task_summary_label)

        self.task_hint_label = QLabel(
            "Tasks can run forward only or use reverse fill to backfill missed rows."
        )
        self.task_hint_label.setObjectName("panelMuted")
        info_layout.addWidget(self.task_hint_label)
        layout.addWidget(info_card)

        self.scene_task_summary_label = QLabel(
            "Incoming scene background tasks will appear here."
        )
        self.scene_task_summary_label.setObjectName("panelPrimary")
        layout.addWidget(self.scene_task_summary_label)

        self.scene_task_list = QListWidget()
        self.scene_task_list.setMaximumHeight(220)
        self.scene_task_list.currentRowChanged.connect(self._update_scene_task_details)
        layout.addWidget(self.scene_task_list)

        self.scene_task_details_label = QLabel("No background scene task selected.")
        self.scene_task_details_label.setObjectName("panelMuted")
        self.scene_task_details_label.setWordWrap(True)
        layout.addWidget(self.scene_task_details_label)

        propagation_label = QLabel("Propagation Tasks")
        propagation_label.setObjectName("panelPrimary")
        layout.addWidget(propagation_label)

        self.task_list = QListWidget()
        self.task_list.currentRowChanged.connect(self._update_task_details)
        layout.addWidget(self.task_list, 1)

        self.task_details_label = QLabel("No task selected.")
        self.task_details_label.setObjectName("panelMuted")
        self.task_details_label.setWordWrap(True)
        layout.addWidget(self.task_details_label)

        self.run_task_button = QPushButton("Run Selected Task")
        self.run_task_button.setEnabled(False)
        self.run_task_button.clicked.connect(self.run_selected_task)
        layout.addWidget(self.run_task_button)

        return panel

    def _build_panel(self, title: str) -> QFrame:
        panel = QFrame()
        panel.setObjectName("sidebarPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        label = QLabel(title)
        label.setObjectName("panelTitle")
        layout.addWidget(label)
        return panel

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow {
                background: #161b22;
                color: #d8e1e8;
            }
            QMenuBar {
                background: #10151c;
                color: #d8e1e8;
                border-bottom: 1px solid #263241;
            }
            QMenuBar::item:selected {
                background: #1c2530;
            }
            QMenu {
                background: #11161d;
                color: #d8e1e8;
                border: 1px solid #263241;
            }
            QMenu::item:selected {
                background: #2d72d9;
            }
            QDockWidget {
                color: #d8e1e8;
            }
            QDockWidget::title {
                background: #10151c;
                padding: 8px 12px;
                border: 1px solid #263241;
                text-align: left;
            }
            QTabWidget::pane {
                background: #10161e;
                border: 1px solid #263241;
                border-top: none;
            }
            QTabBar::tab {
                background: #10151c;
                color: #95a1af;
                padding: 8px 14px;
                border: 1px solid #263241;
                border-bottom: none;
                min-width: 110px;
            }
            QTabBar::tab:selected {
                background: #1a212b;
                color: #f5f7fa;
            }
            #headerBar {
                background: #10151c;
                border-bottom: 1px solid #263241;
            }
            #appTitle {
                color: #f5f7fa;
                font-size: 21px;
                font-weight: 700;
            }
            #appSubtitle {
                color: #8f9cab;
                font-size: 11px;
            }
            #folderSummary {
                color: #cfd9e3;
                padding-left: 12px;
            }
            #toolRail {
                background: #121820;
                border: 1px solid #263241;
                border-radius: 8px;
            }
            #railTitle {
                color: #718091;
                font-size: 11px;
                font-weight: 700;
                padding-left: 2px;
                padding-bottom: 6px;
            }
            #railButton, QPushButton {
                background: #1f2630;
                color: #dde6ee;
                border: 1px solid #2a3442;
                border-radius: 8px;
                padding: 10px 14px;
            }
            #railButton:hover, QPushButton:hover {
                background: #2a3442;
            }
            QPushButton:disabled {
                background: #161b22;
                color: #6d7a88;
                border: 1px solid #28313b;
            }
            #accentButton {
                background: #2d72d9;
                border: none;
                color: white;
                font-weight: 600;
                padding: 10px 16px;
            }
            #accentButton:hover {
                background: #3b83ef;
            }
            #toolToggleButton {
                background: #1b222c;
                color: #cdd7e0;
                border: 1px solid #2a3442;
                border-radius: 8px;
                padding: 10px 12px;
            }
            #toolToggleButton:checked {
                background: #2d72d9;
                color: white;
                border: 1px solid #2d72d9;
            }
            #navBar {
                background: #10151c;
                border: 1px solid #263241;
                border-radius: 8px;
            }
            #counterLabel {
                color: #cfd9e3;
                font-size: 12px;
                font-weight: 700;
            }
            #canvasFrame {
                background: #0f141a;
                border: 1px solid #263241;
                border-radius: 8px;
            }
            #sidebarPanel, #timelineInfoCard {
                background: #11161d;
                border: 1px solid #273241;
                border-radius: 8px;
            }
            #panelTitle {
                color: #cfd9e3;
                font-size: 12px;
                font-weight: 700;
                padding-bottom: 4px;
            }
            #panelPrimary {
                color: #d8e1e8;
            }
            #panelMuted {
                color: #8793a1;
            }
            QListWidget {
                background: #11161d;
                color: #d8e1e8;
                border: none;
                outline: none;
            }
            QListWidget::item {
                padding: 8px 10px;
                border-radius: 6px;
            }
            QListWidget::item:selected {
                background: #2d72d9;
                color: white;
            }
            QCheckBox {
                color: #d8e1e8;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:unchecked {
                border: 1px solid #536170;
                background: #161b22;
                border-radius: 4px;
            }
            QCheckBox::indicator:checked {
                border: 1px solid #2d72d9;
                background: #2d72d9;
                border-radius: 4px;
            }
            QScrollArea {
                border: 1px solid #263241;
                background: #0f141a;
                border-radius: 8px;
            }
            QStatusBar {
                background: #0f141a;
                color: #95a1af;
                border-top: 1px solid #263241;
            }
            """
        )

    def open_folder_dialog(self) -> None:
        start_dir = str(self.folder_path) if self.folder_path else str(Path.home())
        folder = QFileDialog.getExistingDirectory(
            self,
            "Open Image Folder",
            start_dir,
        )
        if not folder:
            return
        self.load_image_folder(Path(folder))

    def import_coco_annotations(self) -> None:
        if not self.image_paths:
            QMessageBox.information(
                self,
                "Open Images First",
                "Load an image folder before importing COCO annotations.",
            )
            return

        start_dir = str(self.folder_path) if self.folder_path else str(Path.home())
        file_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Import COCO Annotations",
            start_dir,
            "JSON Files (*.json)",
        )
        if not file_path:
            return

        try:
            imported_objects, imported_points, skipped_annotations = (
                self._import_coco_annotations_from_path(Path(file_path))
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self,
                "Import Error",
                f"Could not read the COCO file.\n{exc}",
            )
            return

        self.status_bar.showMessage(
            (
                f"Imported {imported_objects} objects and {imported_points} points "
                f"from {Path(file_path).name}"
            )
        )
        QMessageBox.information(
            self,
            "COCO Import Complete",
            (
                f"Imported {imported_objects} objects and {imported_points} points.\n"
                f"Skipped {skipped_annotations} annotation entries that could not be matched or decoded."
            ),
        )

    def _import_coco_annotations_from_path(
        self,
        file_path: Path,
    ) -> tuple[int, int, int]:
        with open(file_path, "r", encoding="utf-8") as handle:
            coco_data = json.load(handle)

        images = coco_data.get("images", [])
        annotations = coco_data.get("annotations", [])
        if not isinstance(images, list) or not isinstance(annotations, list):
            raise ValueError(
                "The selected file does not contain valid COCO-style images and annotations lists."
            )

        loaded_image_lookup = self._loaded_image_lookup()
        category_lookup = {
            category.get("id"): category.get("name", "object")
            for category in coco_data.get("categories", [])
            if isinstance(category, dict)
        }
        image_id_to_path: dict[int, Path] = {}
        image_id_to_size: dict[int, tuple[int, int]] = {}

        for image_info in images:
            if not isinstance(image_info, dict):
                continue
            image_id = image_info.get("id")
            file_name = str(image_info.get("file_name", ""))
            if not isinstance(image_id, int) or not file_name:
                continue
            loaded_path = loaded_image_lookup.get(file_name)
            if loaded_path is None:
                loaded_path = loaded_image_lookup.get(Path(file_name).name)
            if loaded_path is None:
                continue
            image_id_to_path[image_id] = loaded_path
            image_id_to_size[image_id] = self._coerce_image_size(
                image_info.get("width"),
                image_info.get("height"),
                fallback=self._image_size_for_path(loaded_path),
            )

        if not image_id_to_path:
            raise ValueError(
                "None of the COCO image entries match the currently loaded image folder."
            )

        self._record_history_state()
        self.point_annotations = {str(path): [] for path in self.image_paths}
        self.sam2_prompt_points = {str(path): [] for path in self.image_paths}
        self.sam2_prompt_boxes = {str(path): None for path in self.image_paths}
        self.sam2_preview_masks = {str(path): None for path in self.image_paths}
        self.object_annotations = {str(path): [] for path in self.image_paths}
        self.propagation_tasks = []
        self._set_shared_selected_time_window(None)
        self.selected_timeline_objects = set()
        self.selected_timeline_points = set()

        used_point_ids: dict[str, set[int]] = {str(path): set() for path in self.image_paths}
        used_object_ids: dict[str, set[int]] = {str(path): set() for path in self.image_paths}
        imported_points = 0
        imported_objects = 0
        skipped_annotations = 0

        for annotation in annotations:
            if not isinstance(annotation, dict):
                skipped_annotations += 1
                continue
            image_id = annotation.get("image_id")
            if not isinstance(image_id, int):
                skipped_annotations += 1
                continue
            image_path = image_id_to_path.get(image_id)
            if image_path is None:
                skipped_annotations += 1
                continue

            image_key = str(image_path)
            image_size = image_id_to_size.get(image_id) or self._image_size_for_path(image_path)
            if self._is_coco_point_annotation(annotation, category_lookup):
                added_points = self._import_coco_points_for_image(
                    image_key=image_key,
                    annotation=annotation,
                    used_point_ids=used_point_ids[image_key],
                )
                if added_points == 0:
                    skipped_annotations += 1
                imported_points += added_points
                continue

            imported = self._import_coco_object_for_image(
                image_key=image_key,
                annotation=annotation,
                category_lookup=category_lookup,
                image_size=image_size,
                used_object_ids=used_object_ids[image_key],
            )
            if imported:
                imported_objects += 1
            else:
                skipped_annotations += 1

        self._sync_canvas_and_lists()
        self._refresh_task_list()
        return imported_objects, imported_points, skipped_annotations

    def export_coco_annotations(
        self,
        *,
        include_objects: bool = True,
        include_points: bool = True,
    ) -> None:
        if not include_objects and not include_points:
            return
        if not self.image_paths:
            QMessageBox.information(
                self,
                "Open Images First",
                "Load an image folder before exporting COCO annotations.",
            )
            return

        if include_objects and include_points:
            export_title = "Export COCO Annotations"
            default_name = "annotations_coco.json"
            export_kind = "COCO annotations"
        elif include_objects:
            export_title = "Export COCO Objects Only"
            default_name = "annotations_coco_objects.json"
            export_kind = "COCO object annotations"
        else:
            export_title = "Export COCO Keypoints Only"
            default_name = "annotations_coco_keypoints.json"
            export_kind = "COCO keypoint annotations"

        if self.folder_path is not None:
            default_name = str(self.folder_path / default_name)
        file_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            export_title,
            default_name,
            "JSON Files (*.json)",
        )
        if not file_path:
            return
        try:
            self._write_coco_annotations_to_path(
                Path(file_path),
                include_objects=include_objects,
                include_points=include_points,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self,
                "Export Error",
                f"Could not write the COCO file.\n{exc}",
            )
            return

        self.status_bar.showMessage(
            f"Exported {export_kind} to {Path(file_path).name}"
        )

    def _write_coco_annotations_to_path(
        self,
        file_path: Path,
        *,
        include_objects: bool,
        include_points: bool,
    ) -> None:
        coco_payload = self._build_coco_payload(
            include_objects=include_objects,
            include_points=include_points,
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as handle:
            json.dump(coco_payload, handle, indent=2)

    def _build_coco_payload(
        self,
        *,
        include_objects: bool,
        include_points: bool,
    ) -> dict[str, object]:
        category_ids: dict[str, int] = {}
        categories: list[dict[str, object]] = []

        def ensure_category(label: str, supercategory: str) -> int:
            category_id = category_ids.get(label)
            if category_id is not None:
                return category_id
            category_id = len(category_ids) + 1
            category_ids[label] = category_id
            categories.append(
                {
                    "id": category_id,
                    "name": label,
                    "supercategory": supercategory,
                }
            )
            return category_id

        point_category_id = (
            ensure_category("keypoint", "point") if include_points else None
        )
        images_payload: list[dict[str, object]] = []
        annotations_payload: list[dict[str, object]] = []
        annotation_id = 1

        for image_id, image_path in enumerate(self.image_paths, start=1):
            width, height = self._image_size_for_path(image_path)
            images_payload.append(
                {
                    "id": image_id,
                    "file_name": self._image_export_name(image_path),
                    "width": width,
                    "height": height,
                }
            )

            image_key = str(image_path)
            if include_objects:
                for obj in self.object_annotations.get(image_key, []):
                    category_id = ensure_category(obj.label, "object")
                    annotations_payload.append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_id,
                            "annotation_type": "object",
                            "object_id": obj.object_id,
                            "label": obj.label,
                            "bbox": self._bbox_to_coco(obj.bbox),
                            "segmentation": self._encode_mask_to_rle(obj.mask),
                            "area": int(np.count_nonzero(obj.mask)),
                            "iscrowd": 0,
                        }
                    )
                    annotation_id += 1

            if include_points and point_category_id is not None:
                for point in self.point_annotations.get(image_key, []):
                    annotations_payload.append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": point_category_id,
                            "annotation_type": "point",
                            "point_id": point.point_id,
                            "point": [round(point.x, 3), round(point.y, 3)],
                            "keypoints": [round(point.x, 3), round(point.y, 3), 2],
                            "num_keypoints": 1,
                            "bbox": [
                                round(point.x - 2.0, 3),
                                round(point.y - 2.0, 3),
                                4.0,
                                4.0,
                            ],
                            "area": 0,
                            "iscrowd": 0,
                        }
                    )
                    annotation_id += 1

        return {
            "info": {
                "description": "RoboCup Labeling COCO export",
                "version": "1.0",
            },
            "images": images_payload,
            "annotations": annotations_payload,
            "categories": categories,
        }

    def _apply_default_scene_watch_root(self) -> None:
        env_root = os.environ.get("ROBOCUP_SCENE_WATCH_ROOT")
        if not env_root:
            return
        root_path = Path(env_root).expanduser()
        if root_path.exists():
            self._set_scene_watch_root(root_path, start_watch=True)

    def choose_scene_watch_root(self) -> None:
        start_dir = (
            str(self.scene_watch_root)
            if self.scene_watch_root is not None
            else str(Path.home())
        )
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Incoming Scene Root",
            start_dir,
        )
        if not folder:
            return
        self._set_scene_watch_root(Path(folder), start_watch=True)

    def _set_scene_watch_root(self, root: Path, *, start_watch: bool) -> None:
        self.scene_watch_root = root
        self.scene_watch_processed = self._existing_scene_folder_keys(root)
        self.scene_watch_requests = []
        self.scene_watch_list.clear()
        autolabel_mode = _current_autolabeling_mode()
        self.scene_watch_result_label.setText(
            "Detector hook: baseline recorded for "
            f"{len(self.scene_watch_processed)} existing scene folder(s); "
            + (
                "watch root classes.txt will be used for new scenes"
                if autolabel_mode != "tshirt"
                else "tshirt autolabeling mode will use OpenRouter keypoint extraction"
            )
        )
        if start_watch:
            self.scene_watch_timer.start()
            self.toggle_scene_watch_action.setChecked(True)
        else:
            self.scene_watch_timer.stop()
            self.toggle_scene_watch_action.setChecked(False)
        self._update_scene_watch_widgets()
        self.status_bar.showMessage(f"Watching incoming scenes under {root}")
        self._poll_incoming_scene_folders()

    def _toggle_scene_watch_from_action(self, checked: bool) -> None:
        if checked:
            if self.scene_watch_root is None:
                self.toggle_scene_watch_action.setChecked(False)
                self.choose_scene_watch_root()
                return
            self.scene_watch_timer.start()
        else:
            self.scene_watch_timer.stop()
        self._update_scene_watch_widgets()

    def toggle_scene_watch(self) -> None:
        self._toggle_scene_watch_from_action(not self.scene_watch_timer.isActive())

    def scan_incoming_scene_root_now(self) -> None:
        self._poll_incoming_scene_folders()

    def _poll_incoming_scene_folders(self) -> None:
        if self.scene_watch_root is None:
            self._update_scene_watch_widgets()
            return

        autolabel_mode = _current_autolabeling_mode()
        classes_file: Path | None = None
        class_specs: list[VocabularyClassSpec] = []
        if autolabel_mode != "tshirt":
            classes_file = self.scene_watch_root / "classes.txt"
            if not classes_file.exists():
                self.scene_watch_result_label.setText(
                    f"Detector hook: waiting for shared classes file at {classes_file}"
                )
                self._update_scene_watch_widgets()
                return

            class_specs = self._parse_scene_classes_file(classes_file)
            if not class_specs:
                self.scene_watch_result_label.setText(
                    f"Detector hook: shared classes file at {classes_file} is empty or invalid"
                )
                self._update_scene_watch_widgets()
                return

        rgb_root = self.scene_watch_root / "rgb"
        if not rgb_root.exists() or not rgb_root.is_dir():
            self.scene_watch_result_label.setText(
                f"Detector hook: waiting for {rgb_root}"
            )
            self._update_scene_watch_widgets()
            return

        discovered_requests: list[OpenVocabularyDetectionRequest] = []
        for scene_dir in sorted(path for path in rgb_root.iterdir() if path.is_dir()):
            scene_key = str(scene_dir.resolve())
            if scene_key in self.scene_watch_processed:
                continue

            if not self._scene_has_supported_images(scene_dir):
                continue

            request = OpenVocabularyDetectionRequest(
                scene_name=scene_dir.name,
                scene_path=scene_dir,
                image_dir=scene_dir,
                classes_file=classes_file,
                class_specs=copy.deepcopy(class_specs),
            )
            self.scene_watch_processed.add(scene_key)
            self.scene_watch_requests.append(request)
            discovered_requests.append(request)

            if autolabel_mode == "tshirt":
                class_labels = "tshirt keypoints"
            else:
                class_labels = ", ".join(spec.label for spec in class_specs[:4])
                if len(class_specs) > 4:
                    class_labels += f", +{len(class_specs) - 4} more"
            QListWidgetItem(
                f"{request.scene_name}: {class_labels or 'no labels'}",
                self.scene_watch_list,
            )
            while self.scene_watch_list.count() > 20:
                self.scene_watch_list.takeItem(0)

            processing_message = self._enqueue_scene_background_task(request)
            self.scene_watch_result_label.setText(
                f"Detector hook: {processing_message}"
            )

        if discovered_requests:
            self.status_bar.showMessage(
                f"Discovered {len(discovered_requests)} new scene folder(s) under {rgb_root}"
            )

        self._update_scene_watch_widgets()

    def _scene_has_supported_images(self, folder: Path) -> bool:
        try:
            return any(
                path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                for path in folder.iterdir()
            )
        except OSError:
            return False

    def _existing_scene_folder_keys(self, root: Path) -> set[str]:
        rgb_root = root / "rgb"
        if not rgb_root.exists() or not rgb_root.is_dir():
            return set()
        try:
            return {
                str(path.resolve())
                for path in rgb_root.iterdir()
                if path.is_dir()
            }
        except OSError:
            return set()

    def _parse_scene_classes_file(
        self,
        classes_file: Path,
    ) -> list[VocabularyClassSpec]:
        try:
            lines = classes_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []

        class_specs: list[VocabularyClassSpec] = []
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            label = parts[0].strip()
            if not label:
                continue
            description = parts[1].strip() if len(parts) > 1 else label
            class_specs.append(
                VocabularyClassSpec(
                    label=label,
                    description=description,
                )
            )
        return class_specs

    def _update_scene_watch_widgets(self) -> None:
        active = self.scene_watch_timer.isActive()
        self.scene_watch_status_label.setText(
            f"Incoming scene watch: {'on' if active else 'off'}"
        )
        self.scene_watch_path_label.setText(
            f"Watch path: {self.scene_watch_root or '-'}"
        )
        self.toggle_watch_button.setText("Stop Watch" if active else "Start Watch")
        self.toggle_scene_watch_action.setChecked(active)
        has_root = self.scene_watch_root is not None
        self.toggle_watch_button.setEnabled(has_root)
        self.scan_watch_now_button.setEnabled(has_root)
        self.scan_scene_watch_now_action.setEnabled(has_root)
        processed_count = len(self.scene_watch_processed)
        if (
            has_root
            and self.scene_watch_list.count() == 0
            and self.scene_watch_result_label.text() == "Detector hook: idle"
        ):
            self.scene_watch_result_label.setText(
                f"Detector hook: waiting for scene folders ({processed_count} processed)"
            )

    def _enqueue_scene_background_task(
        self,
        request: OpenVocabularyDetectionRequest,
    ) -> str:
        autolabel_mode = _current_autolabeling_mode()
        task_id = 1 if not self.scene_background_tasks else max(
            task.task_id for task in self.scene_background_tasks
        ) + 1
        task = SceneBackgroundTask(
            task_id=task_id,
            request=request,
            output_path=str(self._scene_output_annotations_path(request, autolabel_mode)),
            autolabel_mode=autolabel_mode,
            use_reverse_pass=self.reverse_task_checkbox.isChecked(),
        )
        self.scene_background_tasks.append(task)
        self._refresh_scene_task_list()
        self._start_next_scene_background_task()
        return (
            f"Queued background scene task {task_id} for {request.scene_name}"
            f" ({autolabel_mode})"
        )

    def _start_next_scene_background_task(self) -> None:
        if self.active_scene_task_id is not None:
            return
        next_task = next(
            (task for task in self.scene_background_tasks if task.status == "Queued"),
            None,
        )
        if next_task is None:
            return

        self.active_scene_task_id = next_task.task_id
        next_task.status = "Running"
        next_task.progress = 0
        next_task.message = "Starting background processing"
        self._refresh_scene_task_list()

        worker = threading.Thread(
            target=self._run_scene_background_task_worker,
            args=(copy.deepcopy(next_task),),
            daemon=True,
        )
        self.scene_task_worker_thread = worker
        worker.start()

    def _run_scene_background_task_worker(self, task: SceneBackgroundTask) -> None:
        def report(progress: int, message: str) -> None:
            self.scene_task_update_queue.put(
                SceneTaskUpdate(
                    task_id=task.task_id,
                    status="Running",
                    progress=progress,
                    message=message,
                )
            )

        try:
            if task.autolabel_mode == "tshirt":
                message, output_path = _run_tshirt_scene_processing_pipeline(
                    task,
                    workspace_root=self.workspace_root,
                    package_root=self.sam2_bundle_root,
                    progress_callback=report,
                )
            else:
                message, output_path = _run_scene_processing_pipeline(
                    task,
                    workspace_root=self.workspace_root,
                    package_root=self.sam2_bundle_root,
                    progress_callback=report,
                )
        except Exception as exc:  # noqa: BLE001
            self.scene_task_update_queue.put(
                SceneTaskUpdate(
                    task_id=task.task_id,
                    status="Failed",
                    progress=100,
                    message=str(exc),
                )
            )
            return

        self.scene_task_update_queue.put(
            SceneTaskUpdate(
                task_id=task.task_id,
                status="Completed",
                progress=100,
                message=message,
                output_path=str(output_path),
            )
        )

    def _drain_scene_task_updates(self) -> None:
        updated = False
        while True:
            try:
                update = self.scene_task_update_queue.get_nowait()
            except Empty:
                break

            task = next(
                (scene_task for scene_task in self.scene_background_tasks if scene_task.task_id == update.task_id),
                None,
            )
            if task is None:
                continue
            if update.status is not None:
                task.status = update.status
            if update.progress is not None:
                task.progress = max(0, min(update.progress, 100))
            if update.message is not None:
                task.message = update.message
            if update.output_path is not None:
                task.output_path = update.output_path
            updated = True

            if task.status in {"Completed", "Failed"} and self.active_scene_task_id == task.task_id:
                self.active_scene_task_id = None
                self.scene_task_worker_thread = None
                self.status_bar.showMessage(
                    f"Scene task {task.task_id} {task.status.lower()}"
                )

        if updated:
            self._refresh_scene_task_list()
            self._start_next_scene_background_task()

    def _refresh_scene_task_list(self) -> None:
        selected_row = self.scene_task_list.currentRow()
        self.scene_task_list.clear()
        for row_index, task in enumerate(self.scene_background_tasks):
            item = QListWidgetItem(self.scene_task_list)
            item.setSizeHint(QSize(0, 96))
            widget = SceneTaskListItemWidget(row_index, task)
            widget.load_requested.connect(self._load_scene_task_at_row)
            self.scene_task_list.setItemWidget(item, widget)

        if not self.scene_background_tasks:
            self.scene_task_summary_label.setText(
                "Incoming scene background tasks will appear here."
            )
            self.scene_task_details_label.setText("No background scene task selected.")
            return

        self.scene_task_summary_label.setText(
            f"{len(self.scene_background_tasks)} background scene task(s) queued or completed."
        )
        if selected_row < 0:
            self.scene_task_list.setCurrentRow(len(self.scene_background_tasks) - 1)
        else:
            self.scene_task_list.setCurrentRow(
                min(selected_row, len(self.scene_background_tasks) - 1)
            )
        self._update_scene_task_details(self.scene_task_list.currentRow())

    def _update_scene_task_details(self, row: int) -> None:
        if row < 0 or row >= len(self.scene_background_tasks):
            self.scene_task_details_label.setText("No background scene task selected.")
            return
        task = self.scene_background_tasks[row]
        output_name = Path(task.output_path).name if task.output_path else "-"
        self.scene_task_details_label.setText(
            f"Scene task {task.task_id}: {task.request.scene_name}. "
            f"Mode: {task.autolabel_mode}. "
            f"Status: {task.status}. Output: {output_name}. "
            f"Propagation: {'forward plus reverse fill' if task.use_reverse_pass else 'forward only'}."
        )

    def _load_scene_task_at_row(self, row: int) -> None:
        if row < 0 or row >= len(self.scene_background_tasks):
            return
        task = self.scene_background_tasks[row]
        self.load_image_folder(task.request.image_dir)
        output_path = Path(task.output_path)
        if output_path.exists():
            try:
                self._import_coco_annotations_from_path(output_path)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(
                    self,
                    "Load Scene Output Failed",
                    f"Could not import {output_path.name}.\n{exc}",
                )
                return
        self.status_bar.showMessage(f"Loaded scene {task.request.scene_name}")

    def _scene_output_annotations_path(
        self,
        request: OpenVocabularyDetectionRequest,
        autolabel_mode: str = "objects",
    ) -> Path:
        if self.scene_watch_root is not None:
            output_root = self.scene_watch_root / "output" / request.scene_name
        else:
            output_root = request.scene_path / "output"
        if autolabel_mode == "tshirt":
            return output_root / "annotations_coco_keypoints.json"
        return output_root / "annotations_coco_objects.json"

    def load_image_folder(self, folder: Path) -> None:
        image_paths = sorted(
            path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            QMessageBox.warning(
                self,
                "No Images Found",
                "The selected folder does not contain supported image files.",
            )
            return

        self.folder_path = folder
        self.image_paths = image_paths
        self.current_index = 0
        self._set_shared_selected_time_window(None)
        self.selected_timeline_objects = set()
        self.selected_timeline_points = set()
        self.propagation_tasks = []
        self.point_annotations = {str(path): [] for path in self.image_paths}
        self.sam2_prompt_points = {str(path): [] for path in self.image_paths}
        self.sam2_prompt_boxes = {str(path): None for path in self.image_paths}
        self.sam2_preview_masks = {str(path): None for path in self.image_paths}
        self.object_annotations = {str(path): [] for path in self.image_paths}
        self.undo_stack = []
        self.redo_stack = []
        self._update_history_actions()

        self.file_list.clear()
        for path in self.image_paths:
            QListWidgetItem(path.name, self.file_list)

        self.folder_summary.setText(str(folder))
        self.folder_name_label.setText(f"Folder: {folder.name}")
        self.show_image_at_index(0)
        self._refresh_task_list()
        self.status_bar.showMessage(
            f"Loaded {len(self.image_paths)} images from {folder}"
        )

    def _on_file_selected(self) -> None:
        row = self.file_list.currentRow()
        if row >= 0 and row != self.current_index:
            self.show_image_at_index(row)

    def show_image_at_index(self, index: int) -> None:
        if not self.image_paths or index < 0 or index >= len(self.image_paths):
            return

        image_path = self.image_paths[index]
        reader = QImageReader(str(image_path))
        reader.setAutoTransform(True)
        image = reader.read()
        if image.isNull():
            QMessageBox.critical(
                self,
                "Image Error",
                f"Could not open {image_path.name}.",
            )
            return

        self.current_index = index
        self.current_image = image
        self.current_path = image_path
        self.file_list.blockSignals(True)
        self.file_list.setCurrentRow(index)
        self.file_list.blockSignals(False)
        self.file_list.scrollToItem(self.file_list.item(index))
        self.current_name_label.setText(f"Current image: {image_path.name}")
        self.current_size_label.setText(
            f"Dimensions: {image.width()} x {image.height()}"
        )
        self.image_counter.setText(f"{index + 1} / {len(self.image_paths)}")
        self._sync_canvas_and_lists()
        self.status_bar.showMessage(f"Showing {image_path.name}")

    def add_keypoint(self, x: float, y: float) -> None:
        if self.current_path is None:
            return

        self._record_history_state()
        points = self._current_keypoints()
        next_id = 1 if not points else max(point.point_id for point in points) + 1
        points.append(KeypointAnnotation(point_id=next_id, x=x, y=y))
        self._sync_canvas_and_lists()
        self.status_bar.showMessage(
            f"Added point {next_id} to {self.current_path.name}"
        )

    def clear_current_image_points(self) -> None:
        if self.current_path is None:
            return

        if not self._current_keypoints():
            return
        self._record_history_state()
        self.point_annotations[self._current_image_key()] = []
        self._sync_canvas_and_lists()
        self.status_bar.showMessage(
            f"Cleared points for {self.current_path.name}"
        )

    def add_sam2_prompt_point(self, x: float, y: float) -> None:
        if self.current_path is None:
            return

        self._record_history_state()
        prompts = self._current_sam2_points()
        next_id = 1 if not prompts else max(prompt.prompt_id for prompt in prompts) + 1
        prompts.append(SamPromptPoint(prompt_id=next_id, x=x, y=y))
        self._update_sam2_preview()

    def set_sam2_prompt_box(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> None:
        if self.current_path is None:
            return

        self._record_history_state()
        next_id = self._next_sam2_prompt_id()
        self.sam2_prompt_boxes[self._current_image_key()] = SamPromptBox(
            prompt_id=next_id,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        ).normalized()
        self._update_sam2_preview()

    def clear_current_sam2_prompts(self, *, record_history: bool = True) -> None:
        if self.current_path is None:
            return

        key = self._current_image_key()
        if not self.sam2_prompt_points.get(key) and self.sam2_prompt_boxes.get(key) is None and self.sam2_preview_masks.get(key) is None:
            return
        if record_history:
            self._record_history_state()
        self.sam2_prompt_points[key] = []
        self.sam2_prompt_boxes[key] = None
        self.sam2_preview_masks[key] = None
        self._sync_canvas_and_lists()
        self.status_bar.showMessage(
            f"Cleared SAM2 prompts for {self.current_path.name}"
        )

    def finish_sam2_object(self) -> None:
        if self.current_path is None:
            return

        preview_mask = self._current_preview_mask()
        if preview_mask is None or not np.any(preview_mask):
            QMessageBox.information(
                self,
                "No Preview Mask",
                "Add SAM2 point or box prompts first so a mask can be previewed.",
            )
            return

        label, accepted = QInputDialog.getText(
            self,
            "Finish SAM2 Object",
            "Object label:",
            text=self.last_object_label,
        )
        if not accepted or not label.strip():
            return

        self.last_object_label = label.strip()
        objects = self._current_objects()
        next_id = 1 if not objects else max(obj.object_id for obj in objects) + 1
        bbox = self._mask_bbox(preview_mask)
        if bbox is None:
            QMessageBox.warning(
                self,
                "Invalid Mask",
                "The current preview mask is empty and cannot be finished.",
            )
            return

        self._record_history_state()
        objects.append(
            MaskObject(
                object_id=next_id,
                label=self.last_object_label,
                mask=preview_mask.copy(),
                bbox=bbox,
            )
        )
        self.clear_current_sam2_prompts(record_history=False)
        self.status_bar.showMessage(
            f"Created object #{next_id} ({self.last_object_label})"
        )

    def create_propagation_task_from_selection(self) -> None:
        selected_time_window = self._shared_selected_time_window()
        if selected_time_window is None:
            QMessageBox.information(
                self,
                "No Time Window",
                "Select a frame window in the timeline first.",
            )
            return

        active_object_keys = sorted(self.selected_timeline_objects)
        active_point_keys = sorted(self.selected_timeline_points)
        if not active_object_keys and not active_point_keys:
            QMessageBox.information(
                self,
                "No Active Timeline Rows",
                "Activate at least one object or point row for the selected time window.",
            )
            return

        start_index, end_index = selected_time_window
        frames: list[TaskFrameSelection] = []
        annotated_frame_count = 0

        for frame_index in range(start_index, end_index + 1):
            image_path = self.image_paths[frame_index]
            frame_objects: list[TaskObjectSelection] = []
            frame_points: list[TaskPointSelection] = []
            for obj in self.object_annotations.get(str(image_path), []):
                object_key = self._object_timeline_key(obj)
                if object_key not in self.selected_timeline_objects:
                    continue
                frame_objects.append(
                    TaskObjectSelection(
                        object_key=object_key,
                        object_id=obj.object_id,
                        label=obj.label,
                        bbox=obj.bbox,
                        mask=obj.mask.copy(),
                    )
                )
            for point in self.point_annotations.get(str(image_path), []):
                point_key = self._point_timeline_key(point)
                if point_key not in self.selected_timeline_points:
                    continue
                frame_points.append(
                    TaskPointSelection(
                        point_key=point_key,
                        point_id=point.point_id,
                        x=point.x,
                        y=point.y,
                    )
                )
            if frame_objects or frame_points:
                annotated_frame_count += 1
            frames.append(
                TaskFrameSelection(
                    frame_index=frame_index,
                    image_path=str(image_path),
                    objects=frame_objects,
                    points=frame_points,
                )
            )

        if annotated_frame_count == 0:
            QMessageBox.warning(
                self,
                "No Annotations In Range",
                "The selected objects and points do not have annotations in the chosen frame window yet.",
            )
            return

        self._record_history_state()
        task_id = 1 if not self.propagation_tasks else max(task.task_id for task in self.propagation_tasks) + 1
        self.propagation_tasks.append(
            PropagationTask(
                task_id=task_id,
                start_index=start_index,
                end_index=end_index,
                object_keys=active_object_keys,
                point_keys=active_point_keys,
                frames=frames,
                use_reverse_pass=self.reverse_task_checkbox.isChecked(),
                point_tracker=str(self.point_tracker_selector.currentData() or "sam2"),
            )
        )
        self._refresh_task_list()
        self.bottom_tabs.setCurrentWidget(self.tasks_tab)
        point_tracker_message = ""
        if active_point_keys:
            point_tracker_message = (
                f" using {_point_tracker_display_name(str(self.point_tracker_selector.currentData() or 'sam2'))} for points"
            )
        self.status_bar.showMessage(
            (
                f"Created propagation task {task_id} with "
                f"{len(active_object_keys)} objects and {len(active_point_keys)} points "
                f"across {end_index - start_index + 1} frames"
                f"{point_tracker_message}"
                f"{' using reverse fill' if self.reverse_task_checkbox.isChecked() else ''}"
            )
        )

    def run_selected_task(self) -> None:
        row = self.task_list.currentRow()
        self._run_task_at_row(row)

    def _run_task_at_row(self, row: int) -> None:
        if row < 0 or row >= len(self.propagation_tasks):
            QMessageBox.information(
                self,
                "No Task Selected",
                "Select a propagation task first.",
            )
            return
        if self.folder_path is None:
            QMessageBox.information(
                self,
                "No Image Folder",
                "Load an image folder before running tasks.",
            )
            return

        task = self.propagation_tasks[row]
        previous_status = task.status
        task.status = "Running"
        task.progress = 0
        self._refresh_task_list()
        self.task_list.setCurrentRow(row)
        self.status_bar.showMessage(f"Running propagation task {task.task_id}...")
        QApplication.processEvents()

        try:
            object_results, point_results, model_status = self.sam2_video_engine.run_task(
                folder_path=self.folder_path,
                frame_paths=self.image_paths[task.start_index:task.end_index + 1],
                task=task,
                progress_callback=lambda current, total: self._update_task_progress(
                    row,
                    current,
                    total,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            task.status = "Failed"
            task.progress = 0
            self._refresh_task_list()
            self.task_list.setCurrentRow(row)
            QMessageBox.warning(
                self,
                "Propagation Failed",
                str(exc),
            )
            self.status_bar.showMessage(f"Task {task.task_id} failed")
            return

        task.status = previous_status
        self._record_history_state()
        self._apply_task_propagation_results(
            task=task,
            object_results=object_results,
            point_results=point_results,
        )
        completed_task_id = task.task_id
        self.propagation_tasks.pop(row)
        self._refresh_task_list()
        self._sync_canvas_and_lists()
        self.status_bar.showMessage(
            f"Task {completed_task_id} completed with {model_status}"
        )

    def _update_task_progress(self, row: int, current: int, total: int) -> None:
        if row < 0 or row >= len(self.propagation_tasks):
            return
        progress = 0 if total <= 0 else int((current / total) * 100)
        self.propagation_tasks[row].progress = max(0, min(progress, 100))
        self._refresh_task_list()
        self.task_list.setCurrentRow(row)
        QApplication.processEvents()

    def _apply_task_propagation_results(
        self,
        task: PropagationTask,
        object_results: dict[int, list[MaskObject]],
        point_results: dict[int, list[KeypointAnnotation]],
    ) -> None:
        for frame_index in range(task.start_index, task.end_index + 1):
            image_path = self.image_paths[frame_index]
            image_key = str(image_path)

            if task.object_keys:
                existing_objects = self.object_annotations.get(image_key, [])
                kept_objects = [
                    obj
                    for obj in existing_objects
                    if self._object_timeline_key(obj) not in task.object_keys
                ]
                kept_objects.extend(copy.deepcopy(object_results.get(frame_index, [])))
                self.object_annotations[image_key] = kept_objects

            if task.point_keys:
                existing_points = self.point_annotations.get(image_key, [])
                kept_points = [
                    point
                    for point in existing_points
                    if self._point_timeline_key(point) not in task.point_keys
                ]
                kept_points.extend(copy.deepcopy(point_results.get(frame_index, [])))
                self.point_annotations[image_key] = kept_points

    def delete_selected_objects_in_window(self) -> None:
        if self.timeline_panel_mode == "points":
            self.delete_selected_points_in_window()
            return

        selected_time_window = self._shared_selected_time_window()
        if selected_time_window is None:
            QMessageBox.information(
                self,
                "No Time Window",
                "Select a frame window in the timeline first.",
            )
            return

        active_keys = sorted(self.selected_timeline_objects)
        if not active_keys:
            QMessageBox.information(
                self,
                "No Active Objects",
                "Activate at least one object row for the selected frame window.",
            )
            return

        start_index, end_index = selected_time_window
        deleted_count = 0

        self._record_history_state()
        for frame_index in range(start_index, end_index + 1):
            image_path = self.image_paths[frame_index]
            frame_key = str(image_path)
            objects = self.object_annotations.get(frame_key, [])
            kept_objects: list[MaskObject] = []
            for obj in objects:
                if self._object_timeline_key(obj) in self.selected_timeline_objects:
                    deleted_count += 1
                    continue
                kept_objects.append(obj)
            self.object_annotations[frame_key] = kept_objects

        if deleted_count == 0:
            self._undo_discard_last_record_if_unchanged()
            QMessageBox.information(
                self,
                "Nothing Deleted",
                "The selected object rows do not have finished masks inside that frame window.",
            )
            return

        self.selected_timeline_objects = set()
        self._sync_canvas_and_lists()
        self.status_bar.showMessage(
            f"Deleted {deleted_count} object annotation(s) from frames {start_index + 1} to {end_index + 1}"
        )

    def delete_selected_points_in_window(self) -> None:
        selected_time_window = self._shared_selected_time_window()
        if selected_time_window is None:
            QMessageBox.information(
                self,
                "No Time Window",
                "Select a frame window in the point timeline first.",
            )
            return

        active_keys = sorted(self.selected_timeline_points)
        if not active_keys:
            QMessageBox.information(
                self,
                "No Active Points",
                "Activate at least one point row for the selected frame window.",
            )
            return

        start_index, end_index = selected_time_window
        deleted_count = 0

        self._record_history_state()
        for frame_index in range(start_index, end_index + 1):
            image_path = self.image_paths[frame_index]
            frame_key = str(image_path)
            points = self.point_annotations.get(frame_key, [])
            kept_points: list[KeypointAnnotation] = []
            for point in points:
                if self._point_timeline_key(point) in self.selected_timeline_points:
                    deleted_count += 1
                    continue
                kept_points.append(point)
            self.point_annotations[frame_key] = kept_points

        if deleted_count == 0:
            self._undo_discard_last_record_if_unchanged()
            QMessageBox.information(
                self,
                "Nothing Deleted",
                "The selected point rows do not have annotations inside that frame window.",
            )
            return

        self.selected_timeline_points = set()
        self._sync_canvas_and_lists()
        self.status_bar.showMessage(
            f"Deleted {deleted_count} point annotation(s) from frames {start_index + 1} to {end_index + 1}"
        )

    def clear_timeline_selection(self) -> None:
        self._set_shared_selected_time_window(None)
        self.selected_timeline_objects = set()
        self.selected_timeline_points = set()
        self._refresh_timeline_panel()
        self.status_bar.showMessage("Timeline selection cleared")

    def select_all_timeline_frames(self) -> None:
        if not self.image_paths:
            return
        self._on_timeline_selection_changed(0, len(self.image_paths) - 1)

    def _toggle_bottom_panel(self) -> None:
        self.bottom_dock.setVisible(not self.bottom_dock.isVisible())

    def _set_timeline_panel_mode(self, mode: str) -> None:
        self.timeline_panel_mode = mode
        is_objects = mode == "objects"
        self.object_timeline_button.setChecked(is_objects)
        self.point_timeline_button.setChecked(not is_objects)
        self._refresh_timeline_panel()
        self.status_bar.showMessage(
            f"{'Object' if is_objects else 'Point'} timeline active."
        )

    def _update_sam2_preview(self) -> None:
        if self.current_path is None:
            return

        key = self._current_image_key()
        prompt_points = self._current_sam2_points()
        prompt_box = self._current_sam2_box()
        if not prompt_points and prompt_box is None:
            self.sam2_preview_masks[key] = None
            self._sync_canvas_and_lists()
            return

        try:
            preview_mask, model_status = self.sam2_engine.predict(
                self.current_path,
                prompt_points,
                prompt_box,
            )
            self.sam2_preview_masks[key] = preview_mask
            self.sam2_error_message = None
            self.sam2_model_label.setText(f"Model: {model_status}")
            self.sam2_preview_label.setText(
                "Preview: mask ready"
                if preview_mask is not None and np.any(preview_mask)
                else "Preview: empty"
            )
            self.status_bar.showMessage("SAM2 preview updated")
        except Exception as exc:  # noqa: BLE001
            self.sam2_preview_masks[key] = None
            self.sam2_error_message = str(exc)
            self.sam2_model_label.setText("Model: unavailable")
            self.sam2_preview_label.setText("Preview: failed")
            QMessageBox.warning(
                self,
                "SAM2 Error",
                str(exc),
            )

        self._sync_canvas_and_lists()

    def _sync_canvas_and_lists(self) -> None:
        self.canvas.set_image(self.current_image)
        self.canvas.set_tool_state(self.active_tool, self.sam2_prompt_mode)
        self.canvas.set_keypoints(self._current_keypoints())
        self.canvas.set_show_point_ids(self.show_point_ids)
        self.canvas.set_sam_prompts(
            self._current_sam2_points(),
            self._current_sam2_box(),
        )
        self.canvas.set_preview_mask(self._current_preview_mask())
        self.canvas.set_objects(self._current_objects())
        self._refresh_keypoint_list()
        self._refresh_sam2_prompt_list()
        self._refresh_object_list()
        self._refresh_timeline_panel()

    def _refresh_keypoint_list(self) -> None:
        points = self._current_keypoints()
        self.keypoint_list.clear()
        for point in points:
            QListWidgetItem(
                f"ID {point.point_id}: ({point.x:.1f}, {point.y:.1f})",
                self.keypoint_list,
            )
        self.point_count_label.setText(f"Points in image: {len(points)}")

    def _refresh_sam2_prompt_list(self) -> None:
        points = self._current_sam2_points()
        prompt_box = self._current_sam2_box()
        self.sam2_prompt_list.clear()

        for point in points:
            QListWidgetItem(
                f"Point {point.prompt_id}: ({point.x:.1f}, {point.y:.1f})",
                self.sam2_prompt_list,
            )
        if prompt_box is not None:
            box = prompt_box.normalized()
            QListWidgetItem(
                f"Box {box.prompt_id}: ({box.x1:.1f}, {box.y1:.1f}) -> ({box.x2:.1f}, {box.y2:.1f})",
                self.sam2_prompt_list,
            )

        self.sam2_prompt_count_label.setText(
            f"Prompts: {len(points)} points, {1 if prompt_box is not None else 0} boxes"
        )
        preview_mask = self._current_preview_mask()
        if preview_mask is not None and np.any(preview_mask):
            self.sam2_preview_label.setText("Preview: mask ready")
        elif self.sam2_error_message:
            self.sam2_preview_label.setText("Preview: failed")
        else:
            self.sam2_preview_label.setText("Preview: none")

    def _refresh_object_list(self) -> None:
        objects = self._current_objects()
        self.object_list.clear()
        for obj in objects:
            QListWidgetItem(
                f"#{obj.object_id} {obj.label}",
                self.object_list,
            )
        self.object_count_label.setText(f"Objects in image: {len(objects)}")

    def _edit_object_label_from_item(self, item: QListWidgetItem) -> None:
        if self.current_path is None:
            return

        row = self.object_list.row(item)
        objects = self._current_objects()
        if row < 0 or row >= len(objects):
            return

        obj = objects[row]
        label, accepted = QInputDialog.getText(
            self,
            "Edit Object Label",
            "Object label:",
            text=obj.label,
        )
        if not accepted or not label.strip() or label.strip() == obj.label:
            return

        new_label = label.strip()
        self._record_history_state()
        renamed_count = 0
        for image_path in self.image_paths:
            image_key = str(image_path)
            for frame_obj in self.object_annotations.get(image_key, []):
                if frame_obj.object_id != obj.object_id:
                    continue
                if frame_obj.label != new_label:
                    frame_obj.label = new_label
                    renamed_count += 1

        self.last_object_label = new_label
        self._sync_canvas_and_lists()
        self.status_bar.showMessage(
            f"Renamed object #{obj.object_id} to {new_label} in {renamed_count} annotation(s)"
        )

    def _refresh_timeline_panel(self) -> None:
        is_objects = self.timeline_panel_mode == "objects"
        self.object_timeline_button.setChecked(is_objects)
        self.point_timeline_button.setChecked(not is_objects)

        selected_range = self._shared_selected_time_window()
        if selected_range is not None and self.image_paths:
            start_index, end_index = selected_range
            if start_index >= len(self.image_paths):
                selected_range = None
            else:
                selected_range = (
                    start_index,
                    min(end_index, len(self.image_paths) - 1),
                )
        self._set_shared_selected_time_window(selected_range)

        available_object_keys = self._available_object_keys_in_selected_window()
        available_point_keys = self._available_point_keys_in_selected_window()
        self.selected_timeline_objects &= available_object_keys if selected_range else set()
        self.selected_timeline_points &= available_point_keys if selected_range else set()

        if is_objects:
            active_keys = self.selected_timeline_objects
            available_keys = available_object_keys
            rows = self._build_object_timeline_rows()
            row_header_title = "Objects"
            empty_message = "Finished objects will appear here once you create them."
        else:
            active_keys = self.selected_timeline_points
            available_keys = available_point_keys
            rows = self._build_point_timeline_rows()
            row_header_title = "Points"
            empty_message = "Point annotations will appear here once you create them."

        self.timeline_widget.set_timeline_data(
            frame_paths=self.image_paths,
            rows=rows,
            current_frame_index=self.current_index,
            selected_range=selected_range,
            active_object_keys=active_keys,
            row_header_title=row_header_title,
            empty_message=empty_message,
        )

        if not self.image_paths:
            self.timeline_selection_label.setText(
                "Select a frame window directly on the timeline."
            )
            self.timeline_objects_label.setText(
                "Objects and points can be activated per selected frame range."
            )
            self.create_task_button.setEnabled(False)
            self.delete_timeline_objects_button.setEnabled(False)
            self.clear_timeline_button.setEnabled(False)
            return

        if selected_range is None:
            self.timeline_selection_label.setText(
                "Click and drag across the timeline to choose a frame range."
            )
            row_label = "object" if is_objects else "point"
            self.timeline_objects_label.setText(
                (
                    f"{len(rows)} {row_label} rows available across {len(self.image_paths)} loaded frames. "
                    f"Current task selection: {len(self.selected_timeline_objects)} objects, {len(self.selected_timeline_points)} points."
                )
            )
            self.create_task_button.setEnabled(False)
            self.create_task_button.setText("Create Propagation Task")
            self.create_task_action.setText("Create Propagation Task")
            self.delete_timeline_objects_button.setText(
                "Delete Objects In Window"
                if is_objects
                else "Delete Points In Window"
            )
            self.delete_timeline_objects_action.setText(
                "Delete Selected Objects In Window"
                if is_objects
                else "Delete Selected Points In Window"
            )
            self.delete_timeline_objects_button.setEnabled(False)
            self.clear_timeline_button.setEnabled(False)
            return

        start_index, end_index = selected_range
        frame_count = end_index - start_index + 1
        self.timeline_selection_label.setText(
            f"Selected frames {start_index + 1} to {end_index + 1} ({frame_count} total)."
        )
        subject = "objects" if is_objects else "points"
        active_count = len(active_keys)
        self.timeline_objects_label.setText(
            (
                f"Active {subject} in range: {active_count} of {len(available_keys)} available. "
                f"Task selection currently holds {len(self.selected_timeline_objects)} objects and {len(self.selected_timeline_points)} points."
            )
        )
        self.create_task_button.setText("Create Propagation Task")
        self.create_task_action.setText("Create Propagation Task")
        self.delete_timeline_objects_button.setText(
            "Delete Objects In Window"
            if is_objects
            else "Delete Points In Window"
        )
        self.delete_timeline_objects_action.setText(
            "Delete Selected Objects In Window"
            if is_objects
            else "Delete Selected Points In Window"
        )
        self.create_task_button.setEnabled(
            bool(self.selected_timeline_objects or self.selected_timeline_points)
        )
        self.delete_timeline_objects_button.setEnabled(bool(active_keys))
        self.clear_timeline_button.setEnabled(True)

    def _refresh_task_list(self) -> None:
        selected_row = self.task_list.currentRow()
        self.task_list.clear()
        for row_index, task in enumerate(self.propagation_tasks):
            item = QListWidgetItem(self.task_list)
            item.setSizeHint(QSize(0, 92))
            widget = TaskListItemWidget(row_index, task)
            widget.start_requested.connect(self._run_task_at_row)
            self.task_list.setItemWidget(item, widget)

        if not self.propagation_tasks:
            self.task_summary_label.setText(
                "Tasks created from timeline selections will appear here."
            )
            self.task_details_label.setText("No task selected.")
            self.run_task_button.setEnabled(False)
            return

        self.task_summary_label.setText(
            f"{len(self.propagation_tasks)} propagation task(s) prepared."
        )
        if selected_row < 0:
            self.task_list.setCurrentRow(len(self.propagation_tasks) - 1)
        else:
            self.task_list.setCurrentRow(
                min(selected_row, len(self.propagation_tasks) - 1)
            )

    def _update_task_details(self, row: int) -> None:
        if row < 0 or row >= len(self.propagation_tasks):
            self.task_details_label.setText("No task selected.")
            self.run_task_button.setEnabled(False)
            return

        task = self.propagation_tasks[row]
        object_preview = ", ".join(task.object_keys[:4])
        if len(task.object_keys) > 4:
            object_preview += f", +{len(task.object_keys) - 4} more"
        point_preview = ", ".join(task.point_keys[:4])
        if len(task.point_keys) > 4:
            point_preview += f", +{len(task.point_keys) - 4} more"
        annotated_frames = sum(
            1 for frame in task.frames if frame.objects or frame.points
        )
        self.task_details_label.setText(
            (
                f"Task {task.task_id} spans frames {task.start_index + 1} to {task.end_index + 1}. "
                f"It contains {len(task.object_keys)} selected object rows, "
                f"{len(task.point_keys)} selected point rows, and "
                f"{annotated_frames} frame(s) with stored annotations. "
                f"Mode: {'forward plus reverse fill' if task.use_reverse_pass else 'forward only'}. "
                f"Point tracker: {_point_tracker_display_name(task.point_tracker)}. "
                f"Objects: {object_preview or 'none'}. "
                f"Points: {point_preview or 'none'}."
            )
        )
        self.run_task_button.setEnabled(task.status != "Running")

    def _build_object_timeline_rows(self) -> list[TimelineObjectRow]:
        if not self.image_paths:
            return []

        row_map: dict[str, TimelineObjectRow] = {}
        for frame_index, image_path in enumerate(self.image_paths):
            for obj in self.object_annotations.get(str(image_path), []):
                object_key = self._object_timeline_key(obj)
                row = row_map.get(object_key)
                if row is None:
                    row = TimelineObjectRow(
                        object_key=object_key,
                        label=obj.label,
                        object_id=obj.object_id,
                        visibility=[False] * len(self.image_paths),
                    )
                    row_map[object_key] = row
                row.visibility[frame_index] = True

        return sorted(
            row_map.values(),
            key=lambda row: (row.label.lower(), row.object_id),
        )

    def _build_point_timeline_rows(self) -> list[TimelineObjectRow]:
        if not self.image_paths:
            return []

        row_map: dict[str, TimelineObjectRow] = {}
        for frame_index, image_path in enumerate(self.image_paths):
            for point in self.point_annotations.get(str(image_path), []):
                point_key = self._point_timeline_key(point)
                row = row_map.get(point_key)
                if row is None:
                    row = TimelineObjectRow(
                        object_key=point_key,
                        label="point",
                        object_id=point.point_id,
                        visibility=[False] * len(self.image_paths),
                    )
                    row_map[point_key] = row
                row.visibility[frame_index] = True

        return sorted(
            row_map.values(),
            key=lambda row: row.object_id,
        )

    def _available_object_keys_in_selected_window(self) -> set[str]:
        selected_time_window = self._shared_selected_time_window()
        if selected_time_window is None or not self.image_paths:
            return set()

        start_index, end_index = selected_time_window
        object_keys: set[str] = set()
        for frame_index in range(start_index, end_index + 1):
            image_path = self.image_paths[frame_index]
            for obj in self.object_annotations.get(str(image_path), []):
                object_keys.add(self._object_timeline_key(obj))
        return object_keys

    def _available_point_keys_in_selected_window(self) -> set[str]:
        selected_time_window = self._shared_selected_time_window()
        if selected_time_window is None or not self.image_paths:
            return set()

        start_index, end_index = selected_time_window
        point_keys: set[str] = set()
        for frame_index in range(start_index, end_index + 1):
            image_path = self.image_paths[frame_index]
            for point in self.point_annotations.get(str(image_path), []):
                point_keys.add(self._point_timeline_key(point))
        return point_keys

    def _on_timeline_selection_changed(self, start_index: int, end_index: int) -> None:
        start_index, end_index = sorted((start_index, end_index))
        self._set_shared_selected_time_window((start_index, end_index))
        if self.timeline_panel_mode == "objects":
            available_keys = self._available_object_keys_in_selected_window()
            preserved_keys = self.selected_timeline_objects & available_keys
            self.selected_timeline_objects = (
                preserved_keys if preserved_keys else set(available_keys)
            )
        else:
            available_keys = self._available_point_keys_in_selected_window()
            preserved_keys = self.selected_timeline_points & available_keys
            self.selected_timeline_points = (
                preserved_keys if preserved_keys else set(available_keys)
            )
        self._refresh_timeline_panel()
        self.status_bar.showMessage(
            f"Selected timeline frames {start_index + 1} to {end_index + 1}"
        )

    def _on_timeline_row_toggled(self, row_key: str, active: bool) -> None:
        selected_range = self._shared_selected_time_window()
        if selected_range is None:
            self.status_bar.showMessage("Select a time window before activating timeline rows.")
            return

        active_keys = (
            self.selected_timeline_objects
            if self.timeline_panel_mode == "objects"
            else self.selected_timeline_points
        )
        if active:
            active_keys.add(row_key)
        else:
            active_keys.discard(row_key)
        self._refresh_timeline_panel()
        item_kind = "object" if self.timeline_panel_mode == "objects" else "point"
        self.status_bar.showMessage(
            f"{'Activated' if active else 'Deactivated'} {item_kind} row {row_key} for the selected frame window"
        )

    def undo_last_change(self) -> None:
        if not self.undo_stack:
            return
        self.redo_stack.append(self._capture_app_state())
        state = self.undo_stack.pop()
        self._apply_app_state(state)
        self._update_history_actions()
        self.status_bar.showMessage("Undid last change")

    def redo_last_change(self) -> None:
        if not self.redo_stack:
            return
        self.undo_stack.append(self._capture_app_state())
        state = self.redo_stack.pop()
        self._apply_app_state(state)
        self._update_history_actions()
        self.status_bar.showMessage("Redid last change")

    def _capture_app_state(self) -> AppStateSnapshot:
        return AppStateSnapshot(
            point_annotations=copy.deepcopy(self.point_annotations),
            sam2_prompt_points=copy.deepcopy(self.sam2_prompt_points),
            sam2_prompt_boxes=copy.deepcopy(self.sam2_prompt_boxes),
            sam2_preview_masks=copy.deepcopy(self.sam2_preview_masks),
            object_annotations=copy.deepcopy(self.object_annotations),
            propagation_tasks=copy.deepcopy(self.propagation_tasks),
            current_index=self.current_index,
            timeline_panel_mode=self.timeline_panel_mode,
            selected_object_time_window=self.selected_object_time_window,
            selected_point_time_window=self.selected_point_time_window,
            selected_timeline_objects=set(self.selected_timeline_objects),
            selected_timeline_points=set(self.selected_timeline_points),
        )

    def _apply_app_state(self, state: AppStateSnapshot) -> None:
        self.point_annotations = copy.deepcopy(state.point_annotations)
        self.sam2_prompt_points = copy.deepcopy(state.sam2_prompt_points)
        self.sam2_prompt_boxes = copy.deepcopy(state.sam2_prompt_boxes)
        self.sam2_preview_masks = copy.deepcopy(state.sam2_preview_masks)
        self.object_annotations = copy.deepcopy(state.object_annotations)
        self.propagation_tasks = copy.deepcopy(state.propagation_tasks)
        self.timeline_panel_mode = state.timeline_panel_mode
        self.selected_object_time_window = state.selected_object_time_window
        self.selected_point_time_window = state.selected_point_time_window
        self.selected_timeline_objects = set(state.selected_timeline_objects)
        self.selected_timeline_points = set(state.selected_timeline_points)

        target_index = state.current_index
        if self.image_paths and 0 <= target_index < len(self.image_paths):
            self.show_image_at_index(target_index)
        else:
            self._show_empty_state()
        self._refresh_task_list()

    def _record_history_state(self) -> None:
        self.undo_stack.append(self._capture_app_state())
        if len(self.undo_stack) > self.max_history_depth:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self._update_history_actions()

    def _undo_discard_last_record_if_unchanged(self) -> None:
        if self.undo_stack:
            self.undo_stack.pop()
        self._update_history_actions()

    def _update_history_actions(self) -> None:
        self.undo_action.setEnabled(bool(self.undo_stack))
        self.redo_action.setEnabled(bool(self.redo_stack))

    def _loaded_image_lookup(self) -> dict[str, Path]:
        lookup: dict[str, Path] = {}
        for image_path in self.image_paths:
            lookup[image_path.name] = image_path
            if self.folder_path is not None:
                try:
                    relative_name = str(image_path.relative_to(self.folder_path))
                except ValueError:
                    relative_name = image_path.name
                lookup[relative_name] = image_path
        return lookup

    def _image_export_name(self, image_path: Path) -> str:
        if self.folder_path is None:
            return image_path.name
        try:
            return str(image_path.relative_to(self.folder_path))
        except ValueError:
            return image_path.name

    def _image_size_for_path(self, image_path: Path) -> tuple[int, int]:
        if self.current_path == image_path and self.current_image is not None:
            return self.current_image.width(), self.current_image.height()
        reader = QImageReader(str(image_path))
        size = reader.size()
        if size.isValid():
            return size.width(), size.height()
        image = Image.open(image_path)
        return image.width, image.height

    def _coerce_image_size(
        self,
        width: object,
        height: object,
        fallback: tuple[int, int],
    ) -> tuple[int, int]:
        if isinstance(width, int) and width > 0 and isinstance(height, int) and height > 0:
            return width, height
        return fallback

    def _is_coco_point_annotation(
        self,
        annotation: dict[str, object],
        category_lookup: dict[object, object],
    ) -> bool:
        annotation_type = annotation.get("annotation_type")
        if annotation_type == "point":
            return True
        if annotation.get("point") is not None:
            return True
        keypoints = annotation.get("keypoints")
        if isinstance(keypoints, list) and len(keypoints) >= 3:
            return True
        category_name = category_lookup.get(annotation.get("category_id"))
        return category_name in {"keypoint", "point"}

    def _import_coco_points_for_image(
        self,
        image_key: str,
        annotation: dict[str, object],
        used_point_ids: set[int],
    ) -> int:
        imported_count = 0
        point_candidates: list[tuple[float, float, int | None]] = []

        point_value = annotation.get("point")
        if (
            isinstance(point_value, list)
            and len(point_value) >= 2
            and all(isinstance(value, (int, float)) for value in point_value[:2])
        ):
            preferred_id = annotation.get("point_id")
            point_candidates.append(
                (
                    float(point_value[0]),
                    float(point_value[1]),
                    int(preferred_id) if isinstance(preferred_id, int) else None,
                )
            )
        else:
            keypoints = annotation.get("keypoints")
            if isinstance(keypoints, list):
                for index in range(0, len(keypoints) - 2, 3):
                    x_value, y_value, visibility = keypoints[index:index + 3]
                    if not isinstance(x_value, (int, float)) or not isinstance(y_value, (int, float)):
                        continue
                    if isinstance(visibility, (int, float)) and visibility <= 0:
                        continue
                    preferred_id = annotation.get("point_id")
                    point_candidates.append(
                        (
                            float(x_value),
                            float(y_value),
                            int(preferred_id) if isinstance(preferred_id, int) and len(keypoints) <= 3 else None,
                        )
                    )

        if not point_candidates:
            bbox = annotation.get("bbox")
            if isinstance(bbox, list) and len(bbox) >= 4:
                x_value, y_value, width, height = bbox[:4]
                if all(isinstance(value, (int, float)) for value in (x_value, y_value, width, height)):
                    point_candidates.append(
                        (
                            float(x_value) + float(width) / 2.0,
                            float(y_value) + float(height) / 2.0,
                            None,
                        )
                    )

        for x_value, y_value, preferred_id in point_candidates:
            point_id = self._reserve_local_id(preferred_id, used_point_ids)
            self.point_annotations.setdefault(image_key, []).append(
                KeypointAnnotation(
                    point_id=point_id,
                    x=x_value,
                    y=y_value,
                )
            )
            imported_count += 1

        return imported_count

    def _import_coco_object_for_image(
        self,
        image_key: str,
        annotation: dict[str, object],
        category_lookup: dict[object, object],
        image_size: tuple[int, int],
        used_object_ids: set[int],
    ) -> bool:
        label = str(
            annotation.get("label")
            or category_lookup.get(annotation.get("category_id"))
            or "object"
        )
        mask = self._decode_coco_mask(
            segmentation=annotation.get("segmentation"),
            bbox=annotation.get("bbox"),
            image_size=image_size,
        )
        if mask is None or not np.any(mask):
            return False

        bbox = self._mask_bbox(mask)
        if bbox is None:
            return False

        preferred_id = annotation.get("object_id")
        object_id = self._reserve_local_id(
            int(preferred_id) if isinstance(preferred_id, int) else None,
            used_object_ids,
        )
        self.object_annotations.setdefault(image_key, []).append(
            MaskObject(
                object_id=object_id,
                label=label,
                mask=mask,
                bbox=bbox,
            )
        )
        return True

    def _reserve_local_id(
        self,
        preferred_id: int | None,
        used_ids: set[int],
    ) -> int:
        if preferred_id is not None and preferred_id > 0 and preferred_id not in used_ids:
            used_ids.add(preferred_id)
            return preferred_id

        candidate = 1
        while candidate in used_ids:
            candidate += 1
        used_ids.add(candidate)
        return candidate

    def _decode_coco_mask(
        self,
        segmentation: object,
        bbox: object,
        image_size: tuple[int, int],
    ) -> np.ndarray | None:
        if isinstance(segmentation, dict):
            counts = segmentation.get("counts")
            size = segmentation.get("size")
            if isinstance(counts, list) and isinstance(size, list) and len(size) >= 2:
                return self._decode_rle_mask(counts, int(size[0]), int(size[1]))

        if isinstance(segmentation, list) and segmentation:
            polygon_mask = self._decode_polygon_mask(segmentation, image_size)
            if polygon_mask is not None:
                return polygon_mask

        if isinstance(bbox, list) and len(bbox) >= 4:
            return self._bbox_to_mask(bbox, image_size)

        return None

    def _decode_rle_mask(
        self,
        counts: list[object],
        height: int,
        width: int,
    ) -> np.ndarray | None:
        if height <= 0 or width <= 0:
            return None

        values: list[int] = []
        current_value = 0
        for count in counts:
            if not isinstance(count, int) or count < 0:
                return None
            values.extend([current_value] * count)
            current_value = 1 - current_value

        expected_size = height * width
        if len(values) < expected_size:
            values.extend([0] * (expected_size - len(values)))
        if len(values) != expected_size:
            return None

        flat_mask = np.array(values, dtype=np.uint8)
        return flat_mask.reshape((height, width), order="F").astype(bool)

    def _decode_polygon_mask(
        self,
        segmentation: list[object],
        image_size: tuple[int, int],
    ) -> np.ndarray | None:
        width, height = image_size
        if width <= 0 or height <= 0:
            return None

        mask_image = Image.new("L", (width, height), 0)
        drawer = ImageDraw.Draw(mask_image)
        drew_polygon = False

        for polygon in segmentation:
            if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2 != 0:
                continue
            if not all(isinstance(value, (int, float)) for value in polygon):
                continue
            points = [
                (float(polygon[index]), float(polygon[index + 1]))
                for index in range(0, len(polygon), 2)
            ]
            drawer.polygon(points, fill=1)
            drew_polygon = True

        if not drew_polygon:
            return None
        return np.array(mask_image, dtype=bool)

    def _bbox_to_mask(
        self,
        bbox: list[object],
        image_size: tuple[int, int],
    ) -> np.ndarray | None:
        width, height = image_size
        if width <= 0 or height <= 0:
            return None
        x_value, y_value, box_width, box_height = bbox[:4]
        if not all(isinstance(value, (int, float)) for value in (x_value, y_value, box_width, box_height)):
            return None

        x1 = max(0, min(int(round(float(x_value))), width - 1))
        y1 = max(0, min(int(round(float(y_value))), height - 1))
        x2 = max(x1, min(int(round(float(x_value) + float(box_width) - 1)), width - 1))
        y2 = max(y1, min(int(round(float(y_value) + float(box_height) - 1)), height - 1))
        mask = np.zeros((height, width), dtype=bool)
        mask[y1:y2 + 1, x1:x2 + 1] = True
        return mask

    def _encode_mask_to_rle(self, mask: np.ndarray) -> dict[str, object]:
        flat_mask = mask.astype(np.uint8).flatten(order="F")
        counts: list[int] = []
        current_value = 0
        run_length = 0

        for value in flat_mask:
            if int(value) == current_value:
                run_length += 1
                continue
            counts.append(run_length)
            run_length = 1
            current_value = int(value)
        counts.append(run_length)

        return {
            "size": [int(mask.shape[0]), int(mask.shape[1])],
            "counts": counts,
        }

    def _bbox_to_coco(
        self,
        bbox: tuple[int, int, int, int],
    ) -> list[float]:
        x1, y1, x2, y2 = bbox
        return [
            float(x1),
            float(y1),
            float(x2 - x1 + 1),
            float(y2 - y1 + 1),
        ]

    def _current_image_key(self) -> str:
        return str(self.current_path) if self.current_path is not None else ""

    def _current_keypoints(self) -> list[KeypointAnnotation]:
        if self.current_path is None:
            return []
        return self.point_annotations.setdefault(self._current_image_key(), [])

    def _current_sam2_points(self) -> list[SamPromptPoint]:
        if self.current_path is None:
            return []
        return self.sam2_prompt_points.setdefault(self._current_image_key(), [])

    def _current_sam2_box(self) -> SamPromptBox | None:
        if self.current_path is None:
            return None
        return self.sam2_prompt_boxes.setdefault(self._current_image_key(), None)

    def _current_preview_mask(self) -> np.ndarray | None:
        if self.current_path is None:
            return None
        return self.sam2_preview_masks.setdefault(self._current_image_key(), None)

    def _current_objects(self) -> list[MaskObject]:
        if self.current_path is None:
            return []
        return self.object_annotations.setdefault(self._current_image_key(), [])

    def _next_sam2_prompt_id(self) -> int:
        prompt_ids = [point.prompt_id for point in self._current_sam2_points()]
        current_box = self._current_sam2_box()
        if current_box is not None:
            prompt_ids.append(current_box.prompt_id)
        return 1 if not prompt_ids else max(prompt_ids) + 1

    def _object_timeline_key(self, obj: MaskObject) -> str:
        return f"#{obj.object_id} {obj.label}"

    def _point_timeline_key(self, point: KeypointAnnotation) -> str:
        return f"Point {point.point_id}"

    def _shared_selected_time_window(self) -> tuple[int, int] | None:
        return self.selected_object_time_window or self.selected_point_time_window

    def _set_shared_selected_time_window(
        self,
        selected_range: tuple[int, int] | None,
    ) -> None:
        self.selected_object_time_window = selected_range
        self.selected_point_time_window = selected_range

    def _toggle_point_ids(self, checked: bool) -> None:
        self.show_point_ids = checked
        self.show_ids_action.blockSignals(True)
        self.show_ids_action.setChecked(checked)
        self.show_ids_action.blockSignals(False)
        self.show_ids_checkbox.blockSignals(True)
        self.show_ids_checkbox.setChecked(checked)
        self.show_ids_checkbox.blockSignals(False)
        self.canvas.set_show_point_ids(checked)

    def _set_active_tool(self, tool_name: str) -> None:
        self.active_tool = tool_name
        is_browse = tool_name == "browse"
        is_point = tool_name == "point"
        is_sam2 = tool_name == "sam2"

        self.browse_tool_action.blockSignals(True)
        self.point_tool_action.blockSignals(True)
        self.sam2_tool_action.blockSignals(True)
        self.browse_tool_action.setChecked(is_browse)
        self.point_tool_action.setChecked(is_point)
        self.sam2_tool_action.setChecked(is_sam2)
        self.browse_tool_action.blockSignals(False)
        self.point_tool_action.blockSignals(False)
        self.sam2_tool_action.blockSignals(False)

        self.browse_tool_button.setChecked(is_browse)
        self.point_tool_button.setChecked(is_point)
        self.sam2_tool_button.setChecked(is_sam2)

        self.active_tool_label.setText(
            f"Active tool: {'Browse' if is_browse else 'Point' if is_point else 'SAM2'}"
        )
        self.canvas.set_tool_state(tool_name, self.sam2_prompt_mode)

        if is_point:
            self.status_bar.showMessage(
                "Point tool active. Click on the image to add a keypoint."
            )
        elif is_sam2:
            self.status_bar.showMessage(
                "SAM2 tool active. Add point or box prompts for one object, then click Finish."
            )
        else:
            self.status_bar.showMessage("Browse tool active.")

    def _set_sam2_prompt_mode(self, prompt_mode: str) -> None:
        self.sam2_prompt_mode = prompt_mode
        is_point = prompt_mode == "point"

        self.sam2_point_prompt_action.blockSignals(True)
        self.sam2_box_prompt_action.blockSignals(True)
        self.sam2_point_prompt_action.setChecked(is_point)
        self.sam2_box_prompt_action.setChecked(not is_point)
        self.sam2_point_prompt_action.blockSignals(False)
        self.sam2_box_prompt_action.blockSignals(False)

        self.sam2_point_prompt_button.setChecked(is_point)
        self.sam2_box_prompt_button.setChecked(not is_point)
        self.sam2_mode_label.setText(
            f"Prompt mode: {'Point' if is_point else 'Box'}"
        )
        if self.active_tool != "sam2" and self.current_path is not None:
            self.sam2_model_label.setText("Model: switch to SAM2 tool to place prompts")
        self.canvas.set_tool_state(self.active_tool, self.sam2_prompt_mode)

    def _show_empty_state(self) -> None:
        self.current_image = None
        self.current_path = None
        self._set_shared_selected_time_window(None)
        self.selected_timeline_objects = set()
        self.selected_timeline_points = set()
        self.canvas.set_image(None)
        self.canvas.set_keypoints([])
        self.canvas.set_sam_prompts([], None)
        self.canvas.set_preview_mask(None)
        self.canvas.set_objects([])
        self.image_counter.setText("0 / 0")
        self.current_name_label.setText("Current image: -")
        self.current_size_label.setText("Dimensions: -")
        self.folder_name_label.setText("Folder: -")
        self.keypoint_list.clear()
        self.sam2_prompt_list.clear()
        self.object_list.clear()
        self.point_count_label.setText("Points in image: 0")
        self.sam2_prompt_count_label.setText("Prompts: 0 points, 0 boxes")
        self.sam2_preview_label.setText("Preview: none")
        self.object_count_label.setText("Objects in image: 0")
        self.sam2_model_label.setText("Model: waiting")
        self._refresh_timeline_panel()
        self._refresh_task_list()

    def show_previous_image(self) -> None:
        if not self.image_paths:
            return
        self.show_image_at_index((self.current_index - 1) % len(self.image_paths))

    def show_next_image(self) -> None:
        if not self.image_paths:
            return
        self.show_image_at_index((self.current_index + 1) % len(self.image_paths))

    def _mask_bbox(self, mask: np.ndarray) -> tuple[int, int, int, int] | None:
        positions = np.argwhere(mask)
        if positions.size == 0:
            return None
        y_min, x_min = positions.min(axis=0)
        y_max, x_max = positions.max(axis=0)
        return int(x_min), int(y_min), int(x_max), int(y_max)

    def _show_about(self) -> None:
        QMessageBox.information(
            self,
            "About",
            (
                "RoboCup Labeling\n"
                "A lightweight X-AnyLabeling-inspired starter app with SAM2 prompt segmentation "
                "and a task-selection timeline."
            ),
        )

    def _show_placeholder(self, feature_name: str) -> None:
        QMessageBox.information(
            self,
            "Coming Soon",
            (
                f"{feature_name} is not wired up yet.\n"
                "This starter app currently focuses on image browsing, keypoints, SAM2 prompt segmentation, "
                "and propagation task selection."
            ),
        )
