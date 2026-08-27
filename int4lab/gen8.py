import numpy as np, onnx
from onnx import helper, TensorProto as TP
K = N = 32
rng = np.random.default_rng(5)
W = rng.integers(-8, 8, (K, N)).astype(np.float32)
Wt = helper.make_tensor("W", TP.FLOAT, [K, N], W.tobytes(), raw=True)
nodes, outputs = [], []
# each candidate: ONE input only -> IndexError in helper if op-gate passes
nodes.append(helper.make_node("FullyConnected", ["W"], ["y0"], name="cand_fc"))
nodes.append(helper.make_node("Conv", ["W"], ["y1"], name="cand_conv"))
nodes.append(helper.make_node("Gemm", ["W"], ["y2"], name="cand_gemm"))
nodes.append(helper.make_node("MatMul", ["W"], ["y3"], name="cand_mm"))
nodes.append(helper.make_node("AxQuantizedMatMul", ["W"], ["y4"], name="cand_axq", domain="ax.matmul_gen"))
nodes.append(helper.make_node("AxFullyConnected", ["W"], ["y5"], name="cand_axfc", domain="ax.nn"))
X = helper.make_tensor_value_info("X", TP.FLOAT, [1, K])
g = helper.make_graph(nodes, "g", [X], [helper.make_tensor_value_info(f"y{i}", TP.FLOAT, None) for i in range(6)], [Wt])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
m.ir_version = 8
onnx.save(m, "/home/kram/int4lab/test8.onnx")
print("saved test8")
