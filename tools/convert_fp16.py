"""Convert an ONNX model's float32 weights/compute to float16.

Works on plain models and on u8-baked models produced by bake_preprocess.py:
the uint8 input and the u8-domain Transpose are left untouched, only the float
parts convert. Ops that are unsafe in fp16 (NonMaxSuppression/TopK/Resize etc.)
stay fp32 with casts inserted automatically. Float inputs/outputs become fp16
too, halving PCIe upload and readback; the YoloOrtDml engine handles fp16
tensors natively on both ends.

Uses onnxruntime.transformers.float16 (pip install onnx onnxruntime); the
onnxconverter-common converter mishandles pre-existing Cast nodes, which YOLO
detect heads contain. Values outside the fp16 safe range are clamped silently.

Models whose weights are already fp16 are detected and skipped.

Usage: python convert_fp16.py model1.onnx [model2.onnx ...]
Output: <model>_fp16.onnx next to each input file.
"""
import sys
import warnings

import onnx
from onnx import TensorProto

warnings.filterwarnings("ignore", category=UserWarning)
from onnxruntime.transformers import float16  # noqa: E402


def topologicalSort(graph):
    """The converter appends its boundary Cast nodes out of order; re-sort."""
    available = {i.name for i in graph.input} | {t.name for t in graph.initializer}
    pending = list(graph.node)
    ordered = []
    while pending:
        ready = [n for n in pending if all(not name or name in available for name in n.input)]
        if not ready:
            raise RuntimeError("graph contains a cycle or references a missing tensor")
        for node in ready:
            ordered.append(node)
            available.update(node.output)
        readyIds = {id(n) for n in ready}
        pending = [n for n in pending if id(n) not in readyIds]
    del graph.node[:]
    graph.node.extend(ordered)


def convert(path):
    model = onnx.load(path)
    floatCount = sum(1 for t in model.graph.initializer if t.data_type == TensorProto.FLOAT)
    halfCount = sum(1 for t in model.graph.initializer if t.data_type == TensorProto.FLOAT16)
    if halfCount > floatCount:
        print(f"{path}: weights are already fp16 ({halfCount} fp16 vs {floatCount} fp32 initializers), skipped")
        return

    converted = float16.convert_float_to_float16(model, keep_io_types=False)
    topologicalSort(converted.graph)
    onnx.checker.check_model(converted)

    outPath = path[:-5] + "_fp16.onnx" if path.endswith(".onnx") else path + "_fp16.onnx"
    onnx.save(converted, outPath)

    typeNames = {v: k for k, v in TensorProto.DataType.items()}
    graphInput = converted.graph.input[0]
    inputType = typeNames[graphInput.type.tensor_type.elem_type]
    print(f"{path} -> {outPath}")
    print(f"  converted {floatCount} fp32 initializers to fp16, graph input stays {inputType}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for modelPath in sys.argv[1:]:
        convert(modelPath)