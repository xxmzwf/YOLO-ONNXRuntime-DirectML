"""Bake input preprocessing into a YOLO ONNX model.

Rewrites the model input from float/float16 NCHW to uint8 NHWC (RGB order) and
prepends Cast -> Transpose -> Mul(1/255) nodes, so cast, layout conversion and
normalization run on the GPU. The spatial size is read from the model, so any
input resolution (256/320/640/...) works. Resizing to the model size stays on
the CPU side (see YoloOrtDml preprocess).

Usage: python bake_preprocess.py model1.onnx [model2.onnx ...]
Output: <model>_u8.onnx next to each input file.
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


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


def bake(path):
    model = onnx.load(path)
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

    # transpose while still uint8 so the permute moves 4x fewer bytes
    nodes = [
        helper.make_node("Transpose", [inputName], [nchwOut], name="bake_Transpose", perm=[0, 3, 1, 2]),
        helper.make_node("Cast", [nchwOut], [castOut], name="bake_Cast", to=elemType),
        helper.make_node("Mul", [castOut, scaleName], [original.name], name="bake_Mul"),
    ]
    for node in reversed(nodes):
        graph.node.insert(0, node)

    del graph.input[:]
    graph.input.append(helper.make_tensor_value_info(inputName, TensorProto.UINT8, [1, height, width, channels]))

    # fp16-weight networks produce values already on the fp16 grid, so casting fp32
    # outputs to fp16 halves the GPU->CPU readback without changing any result
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

    onnx.checker.check_model(model)
    outPath = path[:-5] + "_u8.onnx" if path.endswith(".onnx") else path + "_u8.onnx"
    onnx.save(model, outPath)
    networkType = "fp16" if elemType == TensorProto.FLOAT16 else "fp32"
    print(f"{path} -> {outPath}")
    print(f"  input: uint8[1,{height},{width},{channels}] NHWC RGB, network: {networkType}"
          + (f", {castOutputs} output(s) cast to fp16" if castOutputs else ""))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for modelPath in sys.argv[1:]:
        bake(modelPath)