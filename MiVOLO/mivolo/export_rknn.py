"""Convert the face-only MiVOLO ONNX export to an RK3588 RKNN model.
*将 mivolo 的年龄和性别模型由 .onnx 转换为 .rknn
Run in a supported RKNN-Toolkit2 environment (not Toolkit-Lite2).
Default: no INT8 quantization. Runtime input: letterboxed RGB uint8 NHWC.
Normalization is compiled into RKNN; do NOT normalize the runtime image again.

该脚本只能在 linux 系统中运行，且需要安装 rknn-toolkit2
"""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked(ret, stage):
    if ret != 0:
        raise RuntimeError(f"RKNN {stage} failed (return code {ret}). See the Toolkit log above.")


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path,
                        default=PROJECT_ROOT / "mivolo" / "model_onnx" / "mivolo_face_only_rknn.onnx")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "models" / "mivolo_face_only_rk3588.rknn")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-image", type=Path,
                        help="Optional already-cropped BGR-readable face image for simulator/ONNX comparison.")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    model, output = args.onnx.resolve(), args.output.resolve()
    report_path = model.with_suffix(".json")
    output_report = output.with_suffix(".json")
    if not model.is_file() or not report_path.is_file():
        parser.error(f"Missing ONNX or matching JSON: {model}. Run export_onnx.py first.")
    if output.suffix.lower() != ".rknn":
        parser.error("--output must end in .rknn")
    if output_report == report_path:
        parser.error("RKNN metadata would overwrite ONNX metadata; use a different output stem.")
    if not args.check_only:
        for path in (output, output_report):
            if path.exists() and not args.overwrite:
                parser.error(f"Output exists: {path}. Use --overwrite to replace it.")
    if args.verify_image and not args.verify_image.is_file():
        parser.error(f"Test image not found: {args.verify_image}")

    import numpy as np
    import onnx

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("onnx_sha256") != sha256(model):
        raise ValueError("ONNX/JSON checksum mismatch. Re-export or copy the matching pair.")
    if (report.get("mode") != "face_only" or report.get("input_layout") != "NCHW"
            or report.get("input_color") != "RGB" or report.get("input_dtype") != "float32"
            or report.get("output_columns") != ["male_logit", "female_logit", "normalized_age"]):
        raise ValueError("Unsupported ONNX contract. Use the companion export_onnx.py.")
    shape = report["input_shape"]
    if len(shape) != 4 or shape[:2] != [1, 3] or shape[2] != shape[3] or shape[2] <= 0:
        raise ValueError(f"Expected fixed [1, 3, H, H], got {shape}")
    mean, std = np.asarray(report["mean"]), np.asarray(report["std"])
    if (mean.shape != (3,) or std.shape != (3,) or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or not (std > 0).all()):
        raise ValueError("Invalid normalization parameters in JSON.")
    if not np.isfinite([report[k] for k in ("min_age", "max_age", "avg_age")]).all():
        raise ValueError("Invalid age decoding parameters.")
    if report["max_age"] <= report["min_age"]:
        raise ValueError("Invalid age range.")
    graph = onnx.load(str(model))
    onnx.checker.check_model(graph)
    inputs = [v for v in graph.graph.input if v.name not in {x.name for x in graph.graph.initializer}]
    if (len(inputs) != 1 or inputs[0].name != report["input_name"]
            or inputs[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT
            or [d.dim_value for d in inputs[0].type.tensor_type.shape.dim] != shape):
        raise ValueError("Graph input does not match JSON.")
    if (len(graph.graph.output) != 1 or graph.graph.output[0].name != report["output_name"]
            or [d.dim_value for d in graph.graph.output[0].type.tensor_type.shape.dim] != [1, 3]):
        raise ValueError("Graph output does not match the age/gender contract.")
    if any(n.op_type == "Col2Im" for n in graph.graph.node):
        raise ValueError("Col2Im is present; re-export with the companion Fold replacement.")
    print(f"ONNX/metadata checks passed: {model}", flush=True)
    if args.check_only:
        print("File checks only; RKNN compilation and device inference were not tested.")
        return

    try:
        from rknn.api import RKNN
    except ImportError as exc:
        raise SystemExit("RKNN-Toolkit2 is not installed in this Python environment. "
                         "Run this script in your supported Toolkit2 conversion environment; "
                         "Toolkit-Lite2 only runs existing RKNN models.") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    rknn = RKNN(verbose=True)
    try:
        # ONNX expects normalized RGB. Compile normalization for raw RGB 0..255.
        # ONNX: (pixel/255 - mean)/std == RKNN: (pixel - 255*mean)/(255*std).
        checked(rknn.config(target_platform="rk3588", mean_values=[(mean * 255).tolist()],
                            std_values=[(std * 255).tolist()]), "config")
        # The fixed input contract is checked above; no graph cropping is needed.
        checked(rknn.load_onnx(model=str(model)), "load_onnx")
        print("Building for RK3588 without INT8 quantization...", flush=True)
        checked(rknn.build(do_quantization=False), "build")
        verification = {"status": "not_run", "device_inference": "not_tested"}
        if args.verify_image:
            # Self-contained: no torch/timm/ultralytics dependencies on the converter.
            import cv2
            import onnxruntime as ort

            image = cv2.imdecode(np.fromfile(args.verify_image, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Could not read --verify-image.")
            h, w = image.shape[:2]
            size = shape[2]
            ratio = min(size / h, size / w)
            new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            dx, dy = (size - new_w) / 2, (size - new_h) / 2
            image = cv2.copyMakeBorder(image, int(round(dy - .1)), int(round(dy + .1)),
                                       int(round(dx - .1)), int(round(dx + .1)),
                                       cv2.BORDER_CONSTANT, value=(0, 0, 0))
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            normalized = ((rgb / 255.0 - mean) / std).astype(np.float32)
            nchw = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])
            session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
            expected = session.run([report["output_name"]], {report["input_name"]: nchw})[0]
            checked(rknn.init_runtime(), "init_runtime (simulator)")
            results = rknn.inference(inputs=[np.ascontiguousarray(rgb[None])], data_format=["nhwc"])
            if results is None or len(results) != 1:
                raise RuntimeError("RKNN simulator returned no usable output.")
            actual = np.asarray(results[0]).reshape(1, 3)
            if not np.isfinite(actual).all():
                raise ValueError("Non-finite RKNN simulator output.")
            np.testing.assert_allclose(actual, expected, rtol=2e-2, atol=5e-3)
            age_scale = report["max_age"] - report["min_age"]
            verification = {"status": "simulator_passed", "device_inference": "not_tested",
                            "max_abs_error": float(np.max(np.abs(actual - expected))),
                            "age_difference_years": float(abs(actual[0, 2] - expected[0, 2]) * age_scale)}
            print(f"Simulator comparison: {verification}", flush=True)

        with tempfile.TemporaryDirectory(prefix="mivolo_rknn_", dir=str(output.parent)) as temp_dir:
            staged = Path(temp_dir) / output.name
            checked(rknn.export_rknn(str(staged)), "export_rknn")
            if not staged.is_file() or staged.stat().st_size == 0:
                raise RuntimeError("Toolkit did not produce a usable RKNN file.")
            result = {
                "source_onnx": model.name, "source_onnx_sha256": report["onnx_sha256"],
                "rknn_sha256": sha256(staged), "target_platform": "rk3588",
                "do_quantization": False, "runtime_input_layout": "NHWC",
                "runtime_input_shape": [1, shape[2], shape[3], 3],
                "runtime_input_dtype": "uint8", "runtime_input_color": "RGB",
                "runtime_preprocessing": "Crop face, letterbox with black padding, BGR to RGB. "
                                         "Do NOT divide by 255 or normalize; RKNN does it internally.",
                "compiled_mean_values": (mean * 255).tolist(),
                "compiled_std_values": (std * 255).tolist(),
                "output_columns": report["output_columns"],
                "age_decode": report["age_decode"], "gender_decode": report["gender_decode"],
                "min_age": report["min_age"], "max_age": report["max_age"], "avg_age": report["avg_age"],
                "verification": verification,
            }
            staged.replace(output)
            output_report.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Saved: {output}\nMetadata: {output_report}", flush=True)
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
