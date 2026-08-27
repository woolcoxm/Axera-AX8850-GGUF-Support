import numpy as np, onnx
from onnx import helper, TensorProto as TP
K, N = 256, 512
k = np.arange(K)[:, None]; n = np.arange(N)[None, :]
codes = ((3*k + 5*n) % 16 - 8)                       # int4-range codes
nodes, inits, outputs = [], [], []
X = helper.make_tensor_value_info("X", TP.FLOAT, [1, K])

# c0: DequantizeLinear path (QDQ) - Wq int8 Constant with int4-range values
Wq = codes.astype(np.int8)
inits.append(helper.make_tensor("Wq", TP.INT8, [K, N], Wq.tobytes(), raw=True))
inits.append(helper.make_tensor("Ws", TP.FLOAT, [], np.float32(0.5).tobytes(), raw=True))
inits.append(helper.make_tensor("Wz", TP.INT8, [], np.int8(0).tobytes(), raw=True))
nodes.append(helper.make_node("DequantizeLinear", ["Wq", "Ws", "Wz"], ["Wdq"], name="dq_w"))
nodes.append(helper.make_node("MatMul", ["X", "Wdq"], ["y0"], name="mm_after_dq"))
outputs.append(helper.make_tensor_value_info("y0", TP.FLOAT, [1, N]))

# c1: Conv path - fp32 integer int4-range weight [16,8,3,3]
wc = ((np.arange(16*8*3*3).reshape(16,8,3,3) * 7 + 3) % 16 - 8).astype(np.float32)
inits.append(helper.make_tensor("Wc", TP.FLOAT, [16, 8, 3, 3], wc.tobytes(), raw=True))
Xi = helper.make_tensor_value_info("Xi", TP.FLOAT, [1, 8, 16, 16])
nodes.append(helper.make_node("Conv", ["Xi", "Wc"], ["y1"], name="conv0"))
outputs.append(helper.make_tensor_value_info("y1", TP.FLOAT, [1, 16, 14, 14]))

g = helper.make_graph(nodes, "g", [X, Xi], outputs, inits)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
m.ir_version = 8
onnx.save(m, "/home/kram/int4lab/test11.onnx")
print("saved test11 (QDQ + int4-range Conv)")
