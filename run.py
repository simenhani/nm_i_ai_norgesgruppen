import argparse
import json
import os
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE_DIR / ".ultralytics"))

try:
    from ultralytics import YOLO
except Exception as exc:
    raise SystemExit(
        "Failed to import ultralytics. Run `pixi install` and make sure `torchvision` is available.\n"
        f"Original error: {exc}"
    ) from exc

try:
    import numpy as np
except Exception:
    np = None

try:
    from PIL import Image as PILImage
except Exception:
    PILImage = None

METADATA_PATH = BASE_DIR / "category_metadata.json"
RUNS_DIR = BASE_DIR / "runs"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_OUTPUT_NAME = "predictions.json"
DEFAULT_INPUT_CANDIDATES = [
    BASE_DIR / "input",
    BASE_DIR / "inputs",
    BASE_DIR / "test",
    BASE_DIR / "test_images",
    BASE_DIR / "images",
    BASE_DIR / "data",
    Path("/input"),
    Path("/inputs"),
    Path("/data"),
]
DEFAULT_OUTPUT_CANDIDATES = [
    BASE_DIR / DEFAULT_OUTPUT_NAME,
    BASE_DIR / "output" / DEFAULT_OUTPUT_NAME,
    Path("/output") / DEFAULT_OUTPUT_NAME,
]
MODEL_CACHE: dict[str, YOLO] = {}


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO inference on one image or a folder of images.")
    parser.add_argument("input_path", nargs="?", help="Image path or folder with images.")
    parser.add_argument("output_path", nargs="?", help="Where to write predictions.")
    parser.add_argument("--input", dest="input_override", help="Image path or folder with images.")
    parser.add_argument("--output", dest="output_override", help="Where to write predictions.")
    parser.add_argument("--weights", default=None, help="Path to model weights.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS.")
    parser.add_argument("--device", default=None, help="Device override, for example '0' or 'cpu'.")
    return parser.parse_args()


def load_category_metadata(metadata_path: Path) -> dict[int, dict[str, Any]]:
    with metadata_path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    metadata = {}
    for item in data:
        category_id = int(item["category_id"])
        metadata[category_id] = {
            "category_id": category_id,
            "category_name": item.get("category_name") or f"class_{category_id}",
            "product_code": item.get("product_code"),
        }

    return metadata


def find_latest_weights(runs_dir: Path) -> Path | None:
    if not runs_dir.exists():
        return None

    candidates = [path for path in runs_dir.rglob("best.pt") if path.is_file()]
    if not candidates:
        return None

    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_weights_path(weights: str | os.PathLike[str] | None = None) -> Path:
    candidates = []
    if weights:
        candidates.append(Path(weights))

    for env_var in ("MODEL_WEIGHTS", "WEIGHTS_PATH"):
        value = os.environ.get(env_var)
        if value:
            candidates.append(Path(value))

    candidates.extend(
        [
            BASE_DIR / "weights" / "best.pt",
            BASE_DIR / "model" / "best.pt",
            BASE_DIR / "best.pt",
        ]
    )

    latest = find_latest_weights(RUNS_DIR)
    if latest:
        candidates.append(latest)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find model weights. Pass --weights explicitly or place best.pt in weights/, model/, or runs/."
    )


def load_model(weights: str | os.PathLike[str] | None = None) -> YOLO:
    weights_path = resolve_weights_path(weights).resolve()
    cache_key = str(weights_path)

    if cache_key not in MODEL_CACHE:
        MODEL_CACHE[cache_key] = YOLO(str(weights_path))

    return MODEL_CACHE[cache_key]


def _is_path_like(value: Any) -> bool:
    return isinstance(value, (str, os.PathLike))


def _iter_image_paths(directory: Path) -> list[dict[str, Any]]:
    items = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            items.append(
                {
                    "source": str(path),
                    "image_name": path.name,
                    "image_path": str(path.resolve()),
                }
            )
    return items


def normalize_inputs(inputs: Any, start_index: int = 0) -> list[dict[str, Any]]:
    if _is_path_like(inputs):
        input_path = Path(inputs)
        if input_path.is_dir():
            items = _iter_image_paths(input_path)
            if not items:
                raise FileNotFoundError(f"No images found under: {input_path}")
            return items

        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        return [
            {
                "source": str(input_path),
                "image_name": input_path.name,
                "image_path": str(input_path.resolve()),
            }
        ]

    if isinstance(inputs, (list, tuple)):
        items = []
        index = start_index
        for item in inputs:
            normalized = normalize_inputs(item, start_index=index)
            items.extend(normalized)
            index += len(normalized)
        return items

    if np is not None and isinstance(inputs, np.ndarray):
        return [
            {
                "source": inputs,
                "image_name": f"image_{start_index:05d}.png",
                "image_path": None,
            }
        ]

    if PILImage is not None and isinstance(inputs, PILImage.Image):
        name = getattr(inputs, "filename", None)
        image_name = Path(name).name if name else f"image_{start_index:05d}.png"
        image_path = str(Path(name).resolve()) if name else None
        return [
            {
                "source": inputs,
                "image_name": image_name,
                "image_path": image_path,
            }
        ]

    raise TypeError(
        "Unsupported input type. Expected a path, folder, list of paths, numpy array, or PIL image."
    )


def find_default_input() -> Path:
    for env_var in ("INPUT_PATH", "INPUT_DIR", "IMAGE_PATH"):
        value = os.environ.get(env_var)
        if value:
            candidate = Path(value)
            if candidate.exists():
                return candidate

    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not infer an input path. Pass one explicitly with --input or set INPUT_PATH."
    )


def resolve_output_path(output_path: str | os.PathLike[str] | None = None) -> Path:
    if output_path:
        path = Path(output_path)
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            return path

        path.mkdir(parents=True, exist_ok=True)
        return path / DEFAULT_OUTPUT_NAME

    for env_var in ("OUTPUT_PATH", "PREDICTIONS_PATH"):
        value = os.environ.get(env_var)
        if value:
            path = Path(value)
            if path.suffix:
                path.parent.mkdir(parents=True, exist_ok=True)
                return path

            path.mkdir(parents=True, exist_ok=True)
            return path / DEFAULT_OUTPUT_NAME

    for candidate in DEFAULT_OUTPUT_CANDIDATES:
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError:
            continue

    raise FileNotFoundError("Could not infer an output path. Pass one explicitly with --output.")


def serialize_box(box, metadata: dict[int, dict[str, Any]]) -> dict[str, Any]:
    category_id = int(box.cls.item())
    x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
    width = x2 - x1
    height = y2 - y1
    category = metadata.get(
        category_id,
        {
            "category_id": category_id,
            "category_name": f"class_{category_id}",
            "product_code": None,
        },
    )

    return {
        "category_id": category_id,
        "class_id": category_id,
        "category_name": category["category_name"],
        "class_name": category["category_name"],
        "product_name": category["category_name"],
        "product_code": category["product_code"],
        "score": float(box.conf.item()),
        "confidence": float(box.conf.item()),
        "bbox": [x1, y1, width, height],
        "bbox_xywh": [x1, y1, width, height],
        "bbox_xyxy": [x1, y1, x2, y2],
    }


def predict_items(
    model: YOLO,
    items: list[dict[str, Any]],
    metadata: dict[int, dict[str, Any]],
    imgsz: int,
    conf: float,
    iou: float,
    device: str | None,
) -> list[dict[str, Any]]:
    predictions = []

    for item in items:
        kwargs = {
            "source": item["source"],
            "imgsz": imgsz,
            "conf": conf,
            "iou": iou,
            "verbose": False,
        }
        if device:
            kwargs["device"] = device

        result = model.predict(**kwargs)[0]
        detections = [serialize_box(box, metadata) for box in result.boxes]

        predictions.append(
            {
                "image": item["image_name"],
                "file_name": item["image_name"],
                "image_path": item["image_path"],
                "detections": detections,
                "predictions": detections,
            }
        )

    return predictions


def write_predictions(predictions: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".jsonl":
        lines = [json.dumps(item, ensure_ascii=False) for item in predictions]
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        output_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")

    return output_path


def predict(
    inputs: Any,
    weights: str | os.PathLike[str] | None = None,
    imgsz: int = 1280,
    conf: float = 0.25,
    iou: float = 0.7,
    device: str | None = None,
) -> list[dict[str, Any]]:
    items = normalize_inputs(inputs)
    metadata = load_category_metadata(METADATA_PATH)
    model = load_model(weights)
    return predict_items(model, items, metadata, imgsz, conf, iou, device)


def run(
    input_path: Any = None,
    output_path: str | os.PathLike[str] | None = None,
    weights: str | os.PathLike[str] | None = None,
    imgsz: int = 1280,
    conf: float = 0.25,
    iou: float = 0.7,
    device: str | None = None,
) -> list[dict[str, Any]]:
    actual_input = input_path if input_path is not None else find_default_input()
    predictions = predict(actual_input, weights=weights, imgsz=imgsz, conf=conf, iou=iou, device=device)
    destination = resolve_output_path(output_path)
    write_predictions(predictions, destination)
    return predictions


def main():
    args = parse_args()
    input_value = args.input_override or args.input_path
    output_value = args.output_override or args.output_path

    predictions = run(
        input_path=input_value,
        output_path=output_value,
        weights=args.weights,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
    )

    destination = resolve_output_path(output_value)
    print(f"Wrote predictions for {len(predictions)} images to {destination}")


if __name__ == "__main__":
    main()
