"""Convert a YOLO ONNX model for YoloOrtDml.

The conversion is performed entirely in memory:
1. Convert float32 weights and computation to float16.
2. Rewrite the input from float NCHW to uint8 NHWC (RGB order) and bake
   Transpose -> Cast -> Mul(1/255) preprocessing into the graph.

Resizing to the model input size remains on the CPU side. The input model must
have one static NCHW input whose channel count is 1 or 3.

Usage: python onnx_to_yoloortdml.py model1.onnx [model2.onnx ...]
Output: <model>_fp16_u8.onnx next to each input file.
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

warnings.filterwarnings("ignore", category=UserWarning)
from onnxruntime.transformers import float16  # noqa: E402


def log(message):
    print(f"[onnx_to_yoloortdml] {message}", flush=True)


def topologicalSort(graph):
    """The fp16 converter appends boundary Cast nodes out of order; re-sort."""
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


def removeUnusedNodes(graph):
    """Prune a topologically sorted graph, including unused fp16 boundary casts."""
    required = {output.name for output in graph.output}
    retained = []
    for node in reversed(graph.node):
        if required.intersection(node.output):
            retained.append(node)
            required.update(node.input)
    removed = len(graph.node) - len(retained)
    del graph.node[:]
    graph.node.extend(reversed(retained))
    return removed


def pickUnusedName(graph, candidates):
    used = set()
    for node in graph.node:
        used.update(node.input)
        used.update(node.output)
    for value in list(graph.input) + list(graph.output) + list(graph.initializer):
        used.add(value.name)
    for name in candidates:
        if name not in used:
            return name
    raise RuntimeError("no unused tensor name available")


def convertToFp16(model, path):
    floatCount = sum(1 for t in model.graph.initializer if t.data_type == TensorProto.FLOAT)
    halfCount = sum(1 for t in model.graph.initializer if t.data_type == TensorProto.FLOAT16)
    if halfCount > floatCount:
        log(f"{path}: weights are already fp16 ({halfCount} fp16 vs {floatCount} fp32 initializers)")
        return model, 0

    converted = float16.convert_float_to_float16(model, keep_io_types=False)
    topologicalSort(converted.graph)
    return converted, floatCount


def bakePreprocess(model, path):
    graph = model.graph
    if len(graph.input) != 1:
        raise RuntimeError(f"{path}: expected exactly one graph input")

    original = graph.input[0]
    tensorType = original.type.tensor_type
    elemType = tensorType.elem_type
    if elemType not in (TensorProto.FLOAT, TensorProto.FLOAT16):
        raise RuntimeError(f"{path}: input is not float32/float16 (already baked?)")

    dims = [d.dim_value for d in tensorType.shape.dim]
    if len(dims) != 4 or dims[1] not in (1, 3):
        raise RuntimeError(f"{path}: expected NCHW input, got {dims}")
    channels, height, width = dims[1], dims[2], dims[3]
    if height <= 0 or width <= 0:
        raise RuntimeError(f"{path}: dynamic spatial dims are not supported by the bake")

    dtype = np.float16 if elemType == TensorProto.FLOAT16 else np.float32
    scaleName = pickUnusedName(graph, ["bake_scale"])
    graph.initializer.append(numpy_helper.from_array(np.array(1.0 / 255.0, dtype=dtype), name=scaleName))

    inputName = pickUnusedName(graph, ["image", "image_u8", "bake_image"])
    nchwOut = pickUnusedName(graph, ["bake_nchw"])
    castOut = pickUnusedName(graph, ["bake_cast"])

    # Transpose while still uint8 so the permute moves 4x fewer bytes.
    nodes = [
        helper.make_node("Transpose", [inputName], [nchwOut], name="bake_Transpose", perm=[0, 3, 1, 2]),
        helper.make_node("Cast", [nchwOut], [castOut], name="bake_Cast", to=elemType),
        helper.make_node("Mul", [castOut, scaleName], [original.name], name="bake_Mul"),
    ]
    for node in reversed(nodes):
        graph.node.insert(0, node)

    del graph.input[:]
    graph.input.append(helper.make_tensor_value_info(inputName, TensorProto.UINT8, [1, height, width, channels]))

    # FP16-weight networks produce values already on the FP16 grid. Casting FP32
    # outputs to FP16 halves GPU-to-CPU readback without changing the result.
    halfInits = sum(1 for t in graph.initializer if t.data_type == TensorProto.FLOAT16)
    floatInits = sum(1 for t in graph.initializer if t.data_type == TensorProto.FLOAT)
    castOutputs = 0
    if halfInits > floatInits:
        for output in graph.output:
            if output.type.tensor_type.elem_type != TensorProto.FLOAT:
                continue
            name = output.name
            graph.node.append(helper.make_node("Cast", [name], [name + "_fp16"],
                                               name=f"bake_CastOutput{castOutputs}", to=TensorProto.FLOAT16))
            output.name = name + "_fp16"
            output.type.tensor_type.elem_type = TensorProto.FLOAT16
            castOutputs += 1

    return height, width, channels, castOutputs


def convert(path, modelIndex=1, modelCount=1):
    progress = f"[{modelIndex}/{modelCount}]"
    startTime = time.perf_counter()

    log(f"{progress} Loading model: {path}")
    model = onnx.load(path)
    log(f"{progress} Converting weights and graph to fp16 (this may take a while)...")
    model, convertedCount = convertToFp16(model, path)
    log(f"{progress} FP16 conversion complete ({convertedCount} initializer(s) converted)")

    log(f"{progress} Baking uint8 NHWC preprocessing into the graph...")
    height, width, channels, castOutputs = bakePreprocess(model, path)
    log(f"{progress} Preprocessing baked for input {height}x{width}x{channels}")

    log(f"{progress} Re-sorting graph nodes...")
    topologicalSort(model.graph)
    # An unused TopK values output gets a dead Cast during fp16 conversion.
    # Keeping that branch can make DirectML CompileGraph fail and disable fusion.
    removedCount = removeUnusedNodes(model.graph)
    log(f"{progress} Removed {removedCount} unused node(s)")

    log(f"{progress} Checking ONNX model...")
    onnx.checker.check_model(model)

    inputPath = Path(path)
    outputStem = inputPath.stem[:-5] if inputPath.stem.endswith("_fp16") else inputPath.stem
    outPath = inputPath.with_name(outputStem + "_fp16_u8.onnx")

    log(f"{progress} Saving converted model: {outPath}")
    onnx.save(model, outPath)

    log(f"{progress} Done: {path} -> {outPath}")
    log(f"{progress} Input: uint8[1,{height},{width},{channels}] NHWC RGB, network: fp16"
        + (f", {castOutputs} output(s) cast to fp16" if castOutputs else ""))
    log(f"{progress} Elapsed time: {time.perf_counter() - startTime:.2f}s")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    modelPaths = sys.argv[1:]
    modelCount = len(modelPaths)
    log(f"Starting conversion for {modelCount} model(s)")
    for modelIndex, modelPath in enumerate(modelPaths, start=1):
        convert(modelPath, modelIndex, modelCount)
