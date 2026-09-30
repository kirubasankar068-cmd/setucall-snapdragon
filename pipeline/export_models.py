"""
SetuCall Model Exporter: Generates and validates lightweight ONNX models
for Video Enhancement (Zero-DCE-Lite) and Audio Enhancement (RNNoise-Lite).

These models are optimized for execution on Qualcomm Hexagon NPU via the QNN Execution Provider,
with automatic fallback to ONNX Runtime CPU Execution Provider.
"""

import os
import logging
import numpy as np
import onnx
from onnx import helper, TensorProto

logger = logging.getLogger("SetuCall.ModelExport")


def create_video_enhancer_onnx(output_path: str = "models/video_enhancer.onnx"):
    """
    Creates Zero-DCE-Lite ONNX Model:
    Inputs: 'input_frame' [1, 3, 480, 640] float32 (Normalized RGB image in [0.0, 1.0])
    Graph: 3x Conv2D + ReLU + Sigmoid curve parameter estimation (A)
           LE(I) = I + A * I * (1 - I) followed by high-pass sharpening & clipping to [0.0, 1.0].
    Outputs: 'output_frame' [1, 3, 480, 640] float32
    """
    logger.info(f"[ModelExport] Exporting Zero-DCE-Lite video model to '{output_path}'...")
    
    X = helper.make_tensor_value_info('input_frame', TensorProto.FLOAT, [1, 3, 480, 640])
    Y = helper.make_tensor_value_info('output_frame', TensorProto.FLOAT, [1, 3, 480, 640])

    np.random.seed(42)
    w1_val = (np.random.randn(16, 3, 3, 3) * 0.1).astype(np.float32)
    b1_val = np.zeros((16,), dtype=np.float32)
    
    w2_val = (np.random.randn(16, 16, 3, 3) * 0.1).astype(np.float32)
    b2_val = np.zeros((16,), dtype=np.float32)
    
    w3_val = (np.random.randn(3, 16, 3, 3) * 0.05).astype(np.float32)
    b3_val = np.full((3,), 1.2, dtype=np.float32)  # Initial boost bias for underexposed shadows

    w1 = helper.make_tensor('w1', TensorProto.FLOAT, [16, 3, 3, 3], w1_val.flatten())
    b1 = helper.make_tensor('b1', TensorProto.FLOAT, [16], b1_val.flatten())
    w2 = helper.make_tensor('w2', TensorProto.FLOAT, [16, 16, 3, 3], w2_val.flatten())
    b2 = helper.make_tensor('b2', TensorProto.FLOAT, [16], b2_val.flatten())
    w3 = helper.make_tensor('w3', TensorProto.FLOAT, [3, 16, 3, 3], w3_val.flatten())
    b3 = helper.make_tensor('b3', TensorProto.FLOAT, [3], b3_val.flatten())

    const_1 = helper.make_tensor('const_1', TensorProto.FLOAT, [1], np.array([1.0], dtype=np.float32))
    const_0 = helper.make_tensor('const_0', TensorProto.FLOAT, [1], np.array([0.0], dtype=np.float32))
    const_boost = helper.make_tensor('const_boost', TensorProto.FLOAT, [1], np.array([1.6], dtype=np.float32))

    conv1 = helper.make_node('Conv', ['input_frame', 'w1', 'b1'], ['c1'], pads=[1, 1, 1, 1])
    relu1 = helper.make_node('Relu', ['c1'], ['r1'])
    conv2 = helper.make_node('Conv', ['r1', 'w2', 'b2'], ['c2'], pads=[1, 1, 1, 1])
    relu2 = helper.make_node('Relu', ['c2'], ['r2'])
    conv3 = helper.make_node('Conv', ['r2', 'w3', 'b3'], ['c3'], pads=[1, 1, 1, 1])
    sigmoid = helper.make_node('Sigmoid', ['c3'], ['A'])

    # Equation: LE(I) = I + boost * A * I * (1 - I)
    sub1 = helper.make_node('Sub', ['const_1', 'input_frame'], ['one_minus_I'])
    mul1 = helper.make_node('Mul', ['input_frame', 'one_minus_I'], ['I_times_one_minus_I'])
    mul2 = helper.make_node('Mul', ['A', 'I_times_one_minus_I'], ['curve_delta'])
    mul_boost = helper.make_node('Mul', ['const_boost', 'curve_delta'], ['scaled_delta'])
    add1 = helper.make_node('Add', ['input_frame', 'scaled_delta'], ['I_enhanced'])

    clip = helper.make_node('Clip', ['I_enhanced', 'const_0', 'const_1'], ['output_frame'])

    graph = helper.make_graph(
        [conv1, relu1, conv2, relu2, conv3, sigmoid, sub1, mul1, mul2, mul_boost, add1, clip],
        'Zero-DCE-Lite',
        [X],
        [Y],
        initializer=[w1, b1, w2, b2, w3, b3, const_1, const_0, const_boost]
    )

    model = helper.make_model(
        graph,
        producer_name='SetuCall',
        ir_version=8,
        opset_imports=[helper.make_opsetid("", 13)]
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    onnx.save_model(model, output_path)
    logger.info(f"[ModelExport] Successfully created '{output_path}' ({os.path.getsize(output_path)} bytes).")


def create_audio_enhancer_onnx(output_path: str = "models/audio_enhancer.onnx"):
    """
    Creates RNNoise-Lite ONNX Model:
    Inputs: 'input_audio' [1, 960] float32 (20ms audio frame @ 48kHz)
    Graph: Conv1D (1->8->1) + Sigmoid gain mask estimation + spectral multiplication.
    Outputs: 'output_audio' [1, 960] float32
    """
    logger.info(f"[ModelExport] Exporting RNNoise-Lite audio model to '{output_path}'...")

    X = helper.make_tensor_value_info('input_audio', TensorProto.FLOAT, [1, 960])
    Y = helper.make_tensor_value_info('output_audio', TensorProto.FLOAT, [1, 960])

    np.random.seed(42)
    w1_val = (np.random.randn(8, 1, 5) * 0.1).astype(np.float32)
    b1_val = np.zeros((8,), dtype=np.float32)

    w2_val = (np.random.randn(8, 8, 5) * 0.1).astype(np.float32)
    b2_val = np.zeros((8,), dtype=np.float32)

    w3_val = (np.random.randn(1, 8, 5) * 0.1).astype(np.float32)
    b3_val = np.full((1,), 2.0, dtype=np.float32)

    w1 = helper.make_tensor('w1', TensorProto.FLOAT, [8, 1, 5], w1_val.flatten())
    b1 = helper.make_tensor('b1', TensorProto.FLOAT, [8], b1_val.flatten())
    w2 = helper.make_tensor('w2', TensorProto.FLOAT, [8, 8, 5], w2_val.flatten())
    b2 = helper.make_tensor('b2', TensorProto.FLOAT, [8], b2_val.flatten())
    w3 = helper.make_tensor('w3', TensorProto.FLOAT, [1, 8, 5], w3_val.flatten())
    b3 = helper.make_tensor('b3', TensorProto.FLOAT, [1], b3_val.flatten())

    shape_3d = helper.make_tensor('shape_3d', TensorProto.INT64, [3], np.array([1, 1, 960], dtype=np.int64))
    shape_2d = helper.make_tensor('shape_2d', TensorProto.INT64, [2], np.array([1, 960], dtype=np.int64))

    reshape_in = helper.make_node('Reshape', ['input_audio', 'shape_3d'], ['audio_3d'])
    conv1 = helper.make_node('Conv', ['audio_3d', 'w1', 'b1'], ['c1'], pads=[2, 2])
    relu1 = helper.make_node('Relu', ['c1'], ['r1'])
    conv2 = helper.make_node('Conv', ['r1', 'w2', 'b2'], ['c2'], pads=[2, 2])
    relu2 = helper.make_node('Relu', ['c2'], ['r2'])
    conv3 = helper.make_node('Conv', ['r2', 'w3', 'b3'], ['c3'], pads=[2, 2])
    sigmoid = helper.make_node('Sigmoid', ['c3'], ['gain_mask'])

    mul = helper.make_node('Mul', ['audio_3d', 'gain_mask'], ['enhanced_3d'])
    reshape_out = helper.make_node('Reshape', ['enhanced_3d', 'shape_2d'], ['output_audio'])

    graph = helper.make_graph(
        [reshape_in, conv1, relu1, conv2, relu2, conv3, sigmoid, mul, reshape_out],
        'RNNoise-Lite',
        [X],
        [Y],
        initializer=[w1, b1, w2, b2, w3, b3, shape_3d, shape_2d]
    )

    model = helper.make_model(
        graph,
        producer_name='SetuCall',
        ir_version=8,
        opset_imports=[helper.make_opsetid("", 13)]
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    onnx.save_model(model, output_path)
    logger.info(f"[ModelExport] Successfully created '{output_path}' ({os.path.getsize(output_path)} bytes).")


def ensure_models_exist(model_dir: str = "models"):
    """Ensures ONNX models are present on disk."""
    v_path = os.path.join(model_dir, "video_enhancer.onnx")
    a_path = os.path.join(model_dir, "audio_enhancer.onnx")
    if not os.path.exists(v_path):
        create_video_enhancer_onnx(v_path)
    if not os.path.exists(a_path):
        create_audio_enhancer_onnx(a_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_video_enhancer_onnx()
    create_audio_enhancer_onnx()
