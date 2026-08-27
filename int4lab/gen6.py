import numpy as np, onnx
from onnx import helper, TensorProto as TP
K, N = 64, 64
k = np.arange(K)[:, None]; n = np.arange(N)[None, :]
W = ((3*k + 5*n) % 16 - 8).astype(np.float32)
X = helper.make_tensor_value_info("X", TP.FLOAT, [1, K])
Y = helper.make_tensor_value_info("Y", TP.FLOAT, [1, N])
# Malformed: MatMul with ONE input (weight only) -> is_weight_4bits should IndexError
node = helper.make_node("MatMul", ["W"], ["Y"], name="mm_broken")
Wt = helper.make_tensor("W", TP.FLOAT, [K, N], W.tobytes(), raw=True)
g = helper.make_graph([node], "g", [X], [Y], [Wt])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
m.ir_version = 8
onnx.save(m, "/home/kram/int4lab/test6.onnx")
print("saved test6 (1-input MatMul)")
