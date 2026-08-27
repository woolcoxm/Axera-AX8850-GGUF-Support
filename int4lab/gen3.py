import numpy as np, onnx
from onnx import helper, TensorProto as TP

K = N = 64
rng = np.random.default_rng(11)
nodes, inits, inputs, outputs = [], [], [], []

# a0: AxQuantizedMatMul, X int8 input, W int8 INITIALIZER (transposed=1 like the lab's qmm())
inputs.append(helper.make_tensor_value_info("Xq", TP.INT8, [1, K]))
wq = rng.integers(-127, 128, (N, K), dtype=np.int8)
inits.append(helper.make_tensor("W_axq", TP.INT8, [N, K], wq.tobytes(), raw=True))
nodes.append(helper.make_node("AxQuantizedMatMul", ["Xq", "W_axq"], ["Ya"],
    name="axq0", input_scales=(1.0, 1.0), input_zeropoints=(0, 0),
    output_dtype="FP32", output_scales=(1.0,), output_zeropoints=(0,),
    quant_method=0, transposed=1, domain="ax.matmul_gen"))
outputs.append(helper.make_tensor_value_info("Ya", TP.FLOAT, [1, N]))

# a1: MatMul FP16 weights
inputs.append(helper.make_tensor_value_info("Xh", TP.FLOAT16, [1, K]))
wh = (rng.standard_normal((K, N)) * 0.1).astype(np.float16)
inits.append(helper.make_tensor("W_h", TP.FLOAT16, [K, N], wh.tobytes(), raw=True))
nodes.append(helper.make_node("MatMul", ["Xh", "W_h"], ["Yh"], name="a1"))
outputs.append(helper.make_tensor_value_info("Yh", TP.FLOAT16, [1, N]))

# a2: MatMul BF16 weights (stored as FLOAT16 bytes won't work; skip BF16, do DOUBLE small)
# a3: AxQuantizedMatMul with W as graph INPUT (dynamic) - the MoE dream case
inputs.append(helper.make_tensor_value_info("W_dyn", TP.INT8, [N, K]))
nodes.append(helper.make_node("AxQuantizedMatMul", ["Xq", "W_dyn"], ["Yd"],
    name="axq_dyn", input_scales=(1.0, 1.0), input_zeropoints=(0, 0),
    output_dtype="FP32", output_scales=(1.0,), output_zeropoints=(0,),
    quant_method=0, transposed=1, domain="ax.matmul_gen"))
outputs.append(helper.make_tensor_value_info("Yd", TP.FLOAT, [1, N]))

g = helper.make_graph(nodes, "g", inputs, outputs, inits)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 16), helper.make_opsetid("ax.matmul_gen", 1)])
m.ir_version = 8
onnx.save(m, "/home/kram/int4lab/test3.onnx")
print("saved test3.onnx")
