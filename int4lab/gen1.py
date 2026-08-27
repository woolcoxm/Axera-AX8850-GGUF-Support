import numpy as np, onnx
from onnx import helper, TensorProto

K, N = 256, 512
# Deterministic code pattern in [-8,7]: code = (3*k + 5*n) mod 16 - 8, W = code * 0.25
k = np.arange(K)[:, None]; n = np.arange(N)[None, :]
codes = (3*k + 5*n) % 16 - 8
W = (codes.astype(np.float32) * 0.25)
X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, K])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, N])
node = helper.make_node("MatMul", ["X", "W"], ["Y"], name="mm0")
Wt = helper.make_tensor("W", TensorProto.FLOAT, [K, N], W.tobytes(), raw=True)
g = helper.make_graph([node], "g", [X], [Y], [Wt])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
m.ir_version = 8
onnx.checker.check_model(m)
onnx.save(m, "/home/kram/int4lab/test1.onnx")
print("saved test1.onnx W shape", W.shape, "codes sample row0:", codes[0,:12])
