import numpy as np, onnx
from onnx import helper, TensorProto as TP
N, K = 16, 32
idx = np.arange(N*K)
codes = ((idx*7+3) % 16 - 8).astype(np.int8).reshape(N, K, 1, 1)   # 1x1 conv = GEMM [N,K]
Xi = helper.make_tensor_value_info("Xi", TP.FLOAT, [1, K, 1, 1])
y = helper.make_tensor_value_info("y", TP.FLOAT, [1, N, 1, 1])
inits = [
    helper.make_tensor("Wq", TP.INT8, [N,K,1,1], codes.tobytes(), raw=True),
    helper.make_tensor("Ws", TP.FLOAT, [], np.float32(0.5).tobytes(), raw=True),
    helper.make_tensor("Wz", TP.INT8, [], np.int8(0).tobytes(), raw=True),
]
nodes = [
    helper.make_node("DequantizeLinear", ["Wq","Ws","Wz"], ["Wdq"], name="dq_w"),
    helper.make_node("Conv", ["Xi","Wdq"], ["y"], name="conv_gemm"),
]
vi = [helper.make_tensor_value_info("Wdq", TP.FLOAT, [N,K,1,1])]
g = helper.make_graph(nodes, "g", [Xi], [y], inits, value_info=vi)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
m.ir_version = 8
onnx.save(m, "/home/kram/int4lab/conv1x1.onnx")
print("saved conv1x1.onnx [N,K,1,1]=", (N,K))
