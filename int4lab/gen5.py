import numpy as np, onnx
from onnx import helper, TensorProto as TP
K, N = 256, 512
k = np.arange(K)[:, None]; n = np.arange(N)[None, :]
W = ((3*k + 5*n) % 16 - 8).astype(np.float32)   # exact integers in [-8, 7]
X = helper.make_tensor_value_info("X", TP.FLOAT, [1, K])
Y = helper.make_tensor_value_info("Y", TP.FLOAT, [1, N])
node = helper.make_node("MatMul", ["X", "W"], ["Y"], name="mm0")
Wt = helper.make_tensor("W", TP.FLOAT, [K, N], W.tobytes(), raw=True)
g = helper.make_graph([node], "g", [X], [Y], [Wt])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
m.ir_version = 8
onnx.save(m, "/home/kram/int4lab/test5.onnx")
print("saved, W unique vals:", np.unique(W))
