"""Batch test the face-only MiVOLO ONNX model in mivolo_env.

Defaults: input_images -> output_images, using the project's YOLO for face
detection and ONNX Runtime CPU for age/gender. --detector haar uses OpenCV's
cascade; --detector none explicitly treats each image as an already-cropped
face. Preprocessing matches export_onnx.py; no CLAHE or age calibration.
"""

import argparse
import csv
import hashlib
import inspect
import json
from pathlib import Path
import time
from unittest.mock import patch

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=BASE_DIR / "model_onnx/mivolo_face_only_rknn.onnx")
    parser.add_argument("--input-dir", type=Path, default=BASE_DIR / "input_images")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "output_images")
    parser.add_argument("--detector", choices=("yolo", "haar", "none"), default="yolo")
    parser.add_argument("--detector-weights", type=Path,
                        default=PROJECT_ROOT / "models/yolov8x_person_face.pt")
    parser.add_argument("--device", default="cpu", help="YOLO device: cpu, cuda:0, etc. ONNX uses CPU.")
    parser.add_argument("--conf", type=float, default=0.4, help="YOLO face confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO detector image size.")
    parser.add_argument("--threads", type=int, default=4, help="ONNX CPU threads.")
    parser.add_argument("--largest-face", action="store_true", help="Only infer the largest detected face.")
    parser.add_argument("--recursive", action="store_true", help="Include input subdirectories.")
    return parser


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_face(image, size, mean, std):
    """Match mivolo.data.misc.prepare_classification_images exactly."""
    if image is None or image.size == 0:
        raise ValueError("Empty face crop.")
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    new_width, new_height = int(round(width * ratio)), int(round(height * ratio))
    if min(new_width, new_height) == 0:
        raise ValueError("Face crop aspect ratio is too extreme to resize.")
    if (width, height) != (new_width, new_height):
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    dx, dy = (size - new_width) / 2, (size - new_height) / 2
    image = cv2.copyMakeBorder(image, int(round(dy - 0.1)), int(round(dy + 0.1)),
                               int(round(dx - 0.1)), int(round(dx + 0.1)),
                               cv2.BORDER_CONSTANT, value=(0, 0, 0))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    normalized = ((rgb / 255.0 - mean) / std).astype(np.float32)
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])


def decode_prediction(output, metadata):
    output = np.asarray(output)
    if output.shape != (1, 3) or not np.isfinite(output).all():
        raise ValueError("Expected finite ONNX output with shape [1, 3].")
    values = output[0].astype(np.float64)
    probabilities = np.exp(values[:2] - values[:2].max())
    probabilities /= probabilities.sum()
    index = int(probabilities.argmax())
    return {
        "age": float(values[2] * (metadata["max_age"] - metadata["min_age"]) + metadata["avg_age"]),
        "gender": "male" if index == 0 else "female",
        "gender_confidence": float(probabilities[index]),
        "raw_output": values.tolist(),
    }


class AgeGenderONNX:
    def __init__(self, model_path, threads):
        import onnxruntime as ort

        self.path = model_path.resolve()
        self.metadata = json.loads(self.path.with_suffix(".json").read_text(encoding="utf-8"))
        meta = self.metadata
        if meta.get("onnx_sha256") != sha256(self.path):
            raise ValueError("ONNX/JSON checksum mismatch; copy the matching pair from export_onnx.py.")
        if (meta.get("mode") != "face_only" or meta.get("input_color") != "RGB"
                or meta.get("input_layout") != "NCHW" or meta.get("input_dtype") != "float32"
                or meta.get("output_columns") != ["male_logit", "female_logit", "normalized_age"]):
            raise ValueError("Unsupported ONNX metadata; use the companion export_onnx.py.")
        shape = meta["input_shape"]
        if len(shape) != 4 or shape[:2] != [1, 3] or shape[2] != shape[3] or shape[2] <= 0:
            raise ValueError("Expected fixed input [1, 3, H, H].")
        self.size = shape[2]
        self.mean, self.std = np.asarray(meta["mean"]), np.asarray(meta["std"])
        if (self.mean.shape != (3,) or self.std.shape != (3,)
                or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all()
                or not (self.std > 0).all()):
            raise ValueError("Invalid normalization parameters.")
        ages = [meta[key] for key in ("min_age", "max_age", "avg_age")]
        if not np.isfinite(ages).all() or meta["max_age"] <= meta["min_age"]:
            raise ValueError("Invalid age decoding parameters.")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(self.path), options, providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or inputs[0].name != meta["input_name"]
                or inputs[0].shape != shape or inputs[0].type != "tensor(float)"
                or len(outputs) != 1 or outputs[0].name != meta["output_name"]
                or outputs[0].shape != [1, 3]):
            raise ValueError("ONNX input/output does not match JSON metadata.")
        self.predict(np.zeros((self.size, self.size, 3), dtype=np.uint8))

    def predict(self, image):
        tensor = prepare_face(image, self.size, self.mean, self.std)
        start = time.perf_counter()
        output = self.session.run([self.metadata["output_name"]], {self.metadata["input_name"]: tensor})[0]
        elapsed_ms = (time.perf_counter() - start) * 1000
        result = decode_prediction(output, self.metadata)
        result["onnx_ms"] = elapsed_ms
        return result


def clip_box(box, image_shape):
    """Use the project's integer/clipping convention for YOLO face crops."""
    if len(box) != 4 or not np.isfinite(box).all():
        return None
    height, width = image_shape[:2]
    x1, y1, x2, y2 = (int(value) for value in box)
    x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width - 1))
    y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height - 1))
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None


class FaceDetector:
    def __init__(self, args):
        self.args = args
        if args.detector == "yolo":
            if not args.detector_weights.is_file():
                raise FileNotFoundError("YOLO weights not found: " + str(args.detector_weights))
            from ultralytics import YOLO

            if args.detector_weights.suffix.lower() == ".pt":
                # Ultralytics 8.1 uses the old torch.load default. Scope full
                # checkpoint loading to this explicitly selected local .pt only.
                import torch
                from ultralytics.nn import tasks

                weights = args.detector_weights.resolve()
                original = tasks.torch_safe_load

                def load_selected_checkpoint(weight):
                    if Path(weight).resolve() != weights:
                        return original(weight)
                    options = {"weights_only": False} if "weights_only" in inspect.signature(torch.load).parameters else {}
                    return torch.load(str(weights), map_location="cpu", **options), str(weights)

                with patch.object(tasks, "torch_safe_load", load_selected_checkpoint):
                    self.yolo = YOLO(str(weights), task="detect")
            else:
                self.yolo = YOLO(str(args.detector_weights.resolve()), task="detect")
            names = self.yolo.names
            items = names.items() if isinstance(names, dict) else enumerate(names)
            self.face_classes = [int(index) for index, name in items if name.lower() == "face"]
            if not self.face_classes:
                raise ValueError("Detector has no 'face' class; use yolov8x_person_face.pt.")
        elif args.detector == "haar":
            self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            if self.cascade.empty():
                raise RuntimeError("OpenCV face cascade could not be loaded.")

    def detect(self, image):
        args = self.args
        if args.detector == "none":
            height, width = image.shape[:2]
            return [{"bbox": [0, 0, width, height], "face_confidence": None}]
        if args.detector == "haar":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            candidates = [([x, y, x + w, y + h], None) for x, y, w, h in faces]
        else:
            result = self.yolo.predict(source=image, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                                       classes=self.face_classes, device=args.device, half=False,
                                       verbose=False, save=False)[0]
            boxes = result.boxes
            candidates = ([] if boxes is None else
                          list(zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy())))
        faces = []
        for box, score in candidates:
            clipped = clip_box(box, image.shape)
            if clipped is not None:
                faces.append({"bbox": clipped, "face_confidence": None if score is None else float(score)})
        faces.sort(key=lambda face: (face["bbox"][1], face["bbox"][0]))
        if args.largest_face and faces:
            faces = [max(faces, key=lambda face: (face["bbox"][2] - face["bbox"][0])
                         * (face["bbox"][3] - face["bbox"][1]))]
        return faces


def read_image(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix.lower(), image)
    if not success:
        raise OSError("Image encoding failed: " + str(path))
    encoded.tofile(path)


def annotate(image, predictions):
    if not predictions:
        cv2.putText(image, "No face detected", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    for result in predictions:
        x1, y1, x2, y2 = result["bbox"]
        label = "#%d %s %.2f age %.1f" % (result["face_index"], result["gender"],
                                          result["gender_confidence"], result["age"])
        cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), (0, 200, 0), 2)
        font_scale = 0.5
        (width, height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        if width + 4 > image.shape[1]:
            font_scale *= max(1, image.shape[1] - 4) / (width + 2)
            (width, height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        text_y = min(image.shape[0] - baseline - 1, max(height + 3, y1 - 5))
        text_x = max(0, min(x1, image.shape[1] - width - 2))
        cv2.rectangle(image, (text_x, text_y - height - 3),
                      (text_x + width + 2, text_y + baseline), (0, 0, 0), -1)
        cv2.putText(image, label, (text_x + 1, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 1,
                    cv2.LINE_AA)


def write_reports(output_dir, run, records):
    (output_dir / "results.json").write_text(
        json.dumps({"run": run, "images": records}, indent=2, ensure_ascii=False), encoding="utf-8")
    columns = ["image", "status", "face_index", "x1", "y1", "x2", "y2", "face_confidence",
               "age", "gender", "gender_confidence", "onnx_ms", "error"]
    with (output_dir / "results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in records:
            base = {"image": record["image"], "status": record["status"], "error": record.get("error", "")}
            if not record["faces"]:
                writer.writerow(base)
            for face in record["faces"]:
                values = {key: face[key] for key in ("face_index", "face_confidence", "age", "gender",
                                                    "gender_confidence", "onnx_ms")}
                writer.writerow(dict(base, **values, **dict(zip(("x1", "y1", "x2", "y2"), face["bbox"]))))


def main():
    parser = get_parser()
    args = parser.parse_args()
    input_dir, output_dir = args.input_dir.resolve(), args.output_dir.resolve()
    if not input_dir.is_dir():
        parser.error("Input directory not found: " + str(input_dir))
    if input_dir == output_dir or (args.recursive and input_dir in output_dir.parents):
        parser.error("Use a separate output directory outside the recursive input tree.")
    if not 0 < args.conf <= 1 or not 0 < args.iou <= 1 or args.imgsz <= 0 or args.threads <= 0:
        parser.error("conf/iou must be in (0, 1]; imgsz and threads must be positive.")
    files = input_dir.rglob("*") if args.recursive else input_dir.iterdir()
    paths = sorted((path for path in files if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
                   key=lambda path: str(path.relative_to(input_dir)).casefold())
    if not paths:
        parser.error("No supported images found in " + str(input_dir))
    print("Age/gender ONNX:", args.model.resolve(), flush=True)
    predictor = AgeGenderONNX(args.model, args.threads)
    print("Face detector:", args.detector, flush=True)
    detector = FaceDetector(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for number, path in enumerate(paths, 1):
        relative = path.relative_to(input_dir)
        record = {"image": relative.as_posix(), "status": "error", "faces": []}
        start = time.perf_counter()
        try:
            image = read_image(path)
            if image is None:
                raise ValueError("Image could not be decoded.")
            detection_start = time.perf_counter()
            faces = detector.detect(image)
            record["detection_ms"] = (time.perf_counter() - detection_start) * 1000
            record["image_shape"] = list(image.shape)
            for face_index, face in enumerate(faces, 1):
                x1, y1, x2, y2 = face["bbox"]
                prediction = predictor.predict(image[y1:y2, x1:x2])
                record["faces"].append(dict(face, face_index=face_index, **prediction))
            record["status"] = "ok" if faces else "no_face"
            annotated = image.copy()
            annotate(annotated, record["faces"])
            save_image(output_dir / relative, annotated)
            record["output_image"] = relative.as_posix()
            print("[%d/%d] %s: %s, faces=%d" % (number, len(paths), relative, record["status"], len(faces)),
                  flush=True)
            for face in record["faces"]:
                print("  #%d age=%.2f gender=%s confidence=%.3f" %
                      (face["face_index"], face["age"], face["gender"], face["gender_confidence"]), flush=True)
        except Exception as exc:
            record["status"], record["error"] = "error", str(exc)
            print("[%d/%d] %s: ERROR: %s" % (number, len(paths), relative, exc), flush=True)
        record["total_ms"] = (time.perf_counter() - start) * 1000
        records.append(record)
    run = {"age_model": str(predictor.path), "onnx_sha256": predictor.metadata["onnx_sha256"],
           "onnx_provider": "CPUExecutionProvider", "detector": args.detector,
           "detector_weights": str(args.detector_weights.resolve()) if args.detector == "yolo" else None,
           "detector_device": args.device if args.detector == "yolo" else "cpu", "largest_face": args.largest_face,
           "conf": args.conf, "iou": args.iou, "images": len(records),
           "faces": sum(len(record["faces"]) for record in records),
           "no_face_images": sum(record["status"] == "no_face" for record in records),
           "error_images": sum(record["status"] == "error" for record in records)}
    write_reports(output_dir, run, records)
    print("Done: images=%d faces=%d no_face=%d errors=%d" %
          (run["images"], run["faces"], run["no_face_images"], run["error_images"]), flush=True)
    print("Annotated images and results.csv/results.json:", output_dir, flush=True)
    return 1 if run["error_images"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
