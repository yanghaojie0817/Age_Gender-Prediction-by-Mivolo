"""Export the MiVOLO checkpoint directly to an ONNX prepared for RKNN conversion.

Run directly in the IDE, or pass --checkpoint / --output / --overwrite.
Fold is exported as Gather/Add, then onnxsim simplifies the graph. Only the
final mivolo_face_only_rknn.onnx and matching JSON are published.
Detection and image preprocessing run outside this graph. ONNX validation
does not establish RKNN/NPU compatibility.

直接运行：读取 models 中的 MiVOLO 权重，导出用于 RKNN 转换的 ONNX 和 JSON。
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def default_checkpoint():
    # The official download can use 4.22 in its filename although the README
    # labels the linked face/body model 4.24. Preserve the downloaded filename.
    candidates = [PROJECT_ROOT / "models" / name for name in (
        "model_imdb_cross_person_4.22_99.46.pth.tar",
        "model_imdb_cross_person_4.24_99.46.pth.tar",
        "mivolo_imbd.pth.tar",
    )]
    found = [p for p in candidates if p.is_file()]
    return found[0] if len(found) == 1 else candidates[0]


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint())
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "mivolo" / "model_onnx" / "mivolo_face_only_rknn.onnx")
    parser.add_argument("--opset", type=int, default=13,
                        help="Default 13; Fold is exported as fixed Gather/Add overlap sums.")
    parser.add_argument("--simplify", action="store_true",
                        help="Compatibility flag: simplification is now always enabled (requires onnxsim).")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing exports.")
    return parser


def export_fold(tensor, output_size, kernel_size, dilation=1, padding=0, stride=1):
    """Fixed-shape F.fold using Gather/Add, avoiding Col2Im and grouped deconvolution.

    For each output pixel, gather all contributing patch entries and sum them
    in kernel order. Missing contributions point at an appended zero value.
    The RKNN adaptation happens during tracing; no separate preparation step is needed.
    """
    import numpy as np
    import torch

    def pair(value):
        return (value, value) if isinstance(value, int) else tuple(int(v) for v in value)

    height, width = pair(output_size)
    kh, kw = pair(kernel_size)
    dh, dw = pair(dilation)
    ph, pw = pair(padding)
    sh, sw = pair(stride)
    batch, packed_channels, count = (int(v) for v in tensor.shape)
    channels = packed_channels // (kh * kw)
    rows = (height + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    cols = (width + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    if min(height, width, rows, cols) <= 0 or count != rows * cols or packed_channels % (kh * kw):
        raise ValueError("Unexpected Fold shape during fixed-shape export.")

    missing = kh * kw * rows * cols
    contributions = [[] for _ in range(height * width)]
    for ky in range(kh):
        for kx in range(kw):
            for row in range(rows):
                y = row * sh - ph + ky * dh
                if not 0 <= y < height:
                    continue
                for col in range(cols):
                    x = col * sw - pw + kx * dw
                    if 0 <= x < width:
                        contributions[y * width + x].append(
                            (ky * kw + kx) * rows * cols + row * cols + col)
    indices = np.full((max(1, max(map(len, contributions))), height * width), missing, np.int64)
    for pixel, values in enumerate(contributions):
        indices[:len(values), pixel] = values
    flattened = tensor.reshape(batch, channels, missing)
    zero = torch.zeros((batch, channels, 1), dtype=tensor.dtype, device=tensor.device)
    padded = torch.cat((flattened, zero), dim=2)
    total = None
    for values in indices:
        index = torch.tensor(values, dtype=torch.int64, device=tensor.device)
        selected = torch.index_select(padded, 2, index)
        total = selected if total is None else total + selected
    return total.reshape(batch, channels, height, width)


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.opset < 13:
        parser.error("Select opset 13 or higher.")
    checkpoint, output = args.checkpoint.resolve(), args.output.resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}\n"
                     "Download the official age/gender checkpoint, or pass --checkpoint.")
    if output.suffix.lower() != ".onnx":
        parser.error("--output must end in .onnx")
    destinations = [output, output.with_suffix(".json")]
    for path in destinations:
        if path == checkpoint:
            parser.error("Output must not overwrite the checkpoint.")
        if path.exists() and not args.overwrite:
            parser.error(f"Output exists: {path}. Use --overwrite to replace it.")

    # Keep --help and path checks usable even without ML dependencies installed.
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from onnxsim import simplify
    from mivolo.data.misc import prepare_classification_images
    from mivolo.model.mi_volo import MiVOLO

    print(f"Loading: {checkpoint}", flush=True)
    predictor = MiVOLO(str(checkpoint), device="cpu", half=False,
                      use_persons=False, disable_faces=False)
    if predictor.meta.only_age:
        raise ValueError("Select a checkpoint supporting both age and gender.")
    size = predictor.input_size
    mean = [float(v) for v in predictor.data_config["mean"]]
    std = [float(v) for v in predictor.data_config["std"]]
    # Match the original missing-body preprocessing, including normalization.
    # Raw zeros AFTER normalization are not equivalent to a black RGB image.
    missing_body = prepare_classification_images([None], size, mean, std, device="cpu")

    class FaceOnlyWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.core_net = predictor.model
            self.with_persons = predictor.meta.with_persons_model
            self.register_buffer("missing_body", missing_body)

        def forward(self, face_input):
            if self.with_persons:
                # Original channel order: FACE first, BODY second.
                return self.core_net(torch.cat((face_input, self.missing_body), dim=1))
            return self.core_net(face_input)

    wrapper = FaceOnlyWrapper().eval()

    rng = np.random.default_rng(42)
    samples, references = [], []
    with torch.no_grad():
        for height, width in [(size, size), (size + 31, size - 17), (size - 23, size + 47)]:
            image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
            face = prepare_classification_images([image], size, mean, std, device="cpu")
            samples.append(face)
            original_input = (torch.cat((face, missing_body), dim=1)
                              if predictor.meta.with_persons_model else face)
            references.append(predictor.inference(original_input).numpy())

    def verify(path):
        onnx.checker.check_model(str(path))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        errors = []
        for face, expected in zip(samples, references):
            actual = session.run(["output"], {"face_input": face.numpy()})[0]
            if actual.shape != (1, 3) or not np.isfinite(actual).all():
                raise ValueError(f"Invalid output: {actual}")
            np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
            errors.append(float(np.max(np.abs(actual - expected))))
        print(f"Verified {path.name}: max absolute error = {max(errors):.8g}", flush=True)
        return max(errors)

    metadata = {
        "checkpoint": checkpoint.name, "mode": "face_only",
        "source_with_persons_model": bool(predictor.meta.with_persons_model),
        "input_name": "face_input", "input_shape": [1, 3, size, size],
        "input_dtype": "float32", "input_layout": "NCHW", "input_color": "RGB",
        "preprocessing": "Crop face; aspect-preserving bilinear letterbox with black padding; "
                         "BGR to RGB; (pixel / 255 - mean) / std; HWC to NCHW. No CLAHE.",
        "mean": mean, "std": std,
        "output_name": "output", "output_shape": [1, 3],
        "output_columns": ["male_logit", "female_logit", "normalized_age"],
        "gender_decode": "softmax(output[0:2]); index 0 = male, index 1 = female",
        "age_decode": "output[2] * (max_age - min_age) + avg_age",
        "min_age": float(predictor.meta.min_age),
        "max_age": float(predictor.meta.max_age),
        "avg_age": float(predictor.meta.avg_age), "opset": args.opset,
        "fold_export": "fixed Gather/Add overlap sum; no grouped ConvTranspose",
        "simplified": True,
        "rk3588": "ONNX verification does not establish RKNN/NPU compatibility. "
                   "Honor the normalized input contract; do not normalize twice.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Raw and simplified graphs stay temporary; publish only the verified final pair.
    with tempfile.TemporaryDirectory(prefix="mivolo_export_", dir=str(output.parent)) as temp_dir:
        staging = Path(temp_dir)
        raw_path = staging / "raw.onnx"
        options = dict(export_params=True, opset_version=args.opset,
                       do_constant_folding=True, input_names=["face_input"], output_names=["output"])
        signature = inspect.signature(torch.onnx.export).parameters
        if "dynamo" in signature:
            options["dynamo"] = False
        if "external_data" in signature:
            options["external_data"] = False
        print(f"Exporting fixed FP32 input [1, 3, {size}, {size}]...", flush=True)
        with torch.no_grad(), patch("torch.nn.functional.fold", export_fold):
            torch.onnx.export(wrapper, samples[0], str(raw_path), **options)
        print("Simplifying the Gather/Add graph...", flush=True)
        graph, valid = simplify(onnx.load(str(raw_path)))
        if not valid:
            raise RuntimeError("onnxsim validation failed; no export was published.")
        if any(n.op_type in {"Col2Im", "ConvTranspose", "Range"} for n in graph.graph.node):
            raise ValueError("Final graph still contains Col2Im, ConvTranspose or Range; export stopped.")
        path = staging / output.name
        operators = sorted({f"{n.domain or 'ai.onnx'}::{n.op_type}" for n in graph.graph.node})
        report = dict(metadata, operators=operators)
        onnx.helper.set_model_props(graph, {"mivolo": json.dumps(metadata)})
        onnx.save_model(graph, str(path), save_as_external_data=False)
        report["validation_max_abs_error"] = verify(path)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        report["onnx_sha256"] = digest.hexdigest()
        path.replace(output)
        output.with_suffix(".json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved: {output}\nMetadata: {output.with_suffix('.json')}", flush=True)
    print("Done. Run YOLO detection separately before face preprocessing.", flush=True)


if __name__ == "__main__":
    main()
