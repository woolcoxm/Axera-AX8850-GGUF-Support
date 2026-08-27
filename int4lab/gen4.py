import numpy as np, onnx
from onnx import helper, TensorProto as TP

K = N = 1024
rng = np.random.default_rng(3)
nodes, inits, inputs, outputs = [], [], [], []
inputs.append(helper.make_tensor_value_info("input.1", TP.FLOAT, [1, K]))

# c0: big MatMul, torch-export-style names
w0 = (rng.standard_normal((K, N)) * 0.02).astype(np.float32)
inits.append(helper.make_tensor("onnx::MatMul_1001", TP.FLOAT, [K, N], w0.tobytes(), raw=True))
nodes.append(helper.make_node("MatMul", ["input.1", "onnx::MatMul_1001"], ["c0_out"], name="/model/layers.0/mlp/up_proj/MatMul"))
outputs.append(helper.make_tensor_value_info("c0_out", TP.FLOAT, [1, N]))

# c1: full QDQ quantized MatMul (weight + activation both dequantized from int8)
xq = helper.make_tensor_value_info("xq_c1", TP.INT8, [1, K])  # not used as graph input; keep internal
wq1 = rng.integers(-127, 128, (K, N), dtype=np.int8)
inits.append(helper.make_tensor("wq_c1", TP.INT8, [K, N], wq1.tobytes(), raw=True))
inits.append(helper.make_tensor("ws_c1", TP.FLOAT, [N], (np.full(N, 0.01, np.float32)).tobytes(), raw=True))
inits.append(helper.make_tensor("wz_c1", TP.INT8, [N], (np.zeros(N, np.int8)).tobytes(), raw=True))
nodes.append(helper.make_node("DequantizeLinear", ["wq_c1", "ws_c1", "wz_c1"], ["wdq_c1"], axis=1, name="dq_w_c1"))
nodes.append(helper.make_node("MatMul", ["input.1", "wdq_c1"], ["c1_out"], name="c1_matmul"))
outputs.append(helper.make_tensor_value_info("c1_out", TP.FLOAT, [1, N]))

# c2: QLinearMatMul (classic int8 op, opset 10+, works in 13)
inits.append(helper.make_tensor("wq_c2", TP.INT8, [K, N], wq1.tobytes(), raw=True))  # reuse pattern
inits.append(helper.make_tensor("xs_c2", TP.FLOAT, [], np.float32(0.5).tobytes(), raw=True))
inits.append(helper.make_tensor("xz_c2", TP.INT8, [], np.int8(0).tobytes(), raw=True))
inits.append(helper.make_tensor("ys_c2", TP.FLOAT, [], np.float32(0.1).tobytes(), raw=True))
inits.append(helper.make_tensor("yz_c2", TP.INT8, [], np.int8(0).tobytes(), raw=True))
# pre-quantized activation as internal: make x via QuantizeLinear of input.1
nodes.append(helper.make_node("QuantizeLinear", ["input.1", "xs_c2", "xz_c2"], ["xq_c2"], name="q_x_c2"))
nodes.append(helper.make_node("QLinearMatMul", ["xq_c2", "xs_c2", "xz_c2", "wq_c2", "ws_c1", "wz_c1", "ys_c2", "yz_c2"], ["yq_c2"], name="qlmm_c2"))
nodes.append(helper.make_node("DequantizeLinear", ["yq_c2", "ys_c2", "yz_c2"], ["c2_out"], name="dq_y_c2"))
outputs.append(helper.make_tensor_value_info("c2_out", TP.FLOAT, [1, N]))

g = helper.make_graph(nodes, "g", inputs, outputs, inits)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
m.ir_version = 8
m.producer_name = "pytorch"; m.producer_version = "2.4.1"
onnx.save(m, "/home/kram/int4lab/test4.onnx")
print("saved test4.onnx")
