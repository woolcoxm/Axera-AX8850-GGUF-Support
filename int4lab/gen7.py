import numpy as np, onnx
from onnx import helper, TensorProto as TP
K, N = 64, 64
X = helper.make_tensor_value_info("X", TP.FLOAT, [1, K])
Wd = helper.make_tensor_value_info("W", TP.FLOAT, [K, N])   # dynamic weight INPUT
Y = helper.make_tensor_value_info("Y", TP.FLOAT, [1, N])
node = helper.make_node("MatMul", ["X", "W"], ["Y"], name="mm_dyn")
g = helper.make_graph([node], "g", [X, Wd], [Y])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
m.ir_version = 8
onnx.save(m, "/home/kram/int4lab/test7.onnx")
print("saved test7 (weight is graph input)")
