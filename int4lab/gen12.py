import numpy as np, onnx
from onnx import helper, TensorProto as TP
K, N = 256, 512
k = np.arange(K)[:, None]; n = np.arange(N)[None, :]
codes = ((3*k + 5*n) % 16 - 8)
nodes, inits, inputs, outputs, vinfo = [], [], [], [], []
X = helper.make_tensor_value_info("X", TP.FLOAT, [1, K])
inputs.append(X)

def make_branch(tag, w_dtype, with_vinfo):
    Wq = codes.astype(np.int8) if w_dtype == TP.INT8 else codes.astype(np.float32)
    inits.append(helper.make_tensor(f"Wq_{tag}", w_dtype, [K, N], Wq.tobytes(), raw=True))
    inits.append(helper.make_tensor(f"Ws_{tag}", TP.FLOAT, [], np.float32(0.5).tobytes(), raw=True))
    inits.append(helper.make_tensor(f"Wz_{tag}", TP.INT8, [], np.int8(0).tobytes(), raw=True))
    nodes.append(helper.make_node("DequantizeLinear", [f"Wq_{tag}", f"Ws_{tag}", f"Wz_{tag}"], [f"Wdq_{tag}"], name=f"dq_{tag}"))
    nodes.append(helper.make_node("MatMul", ["X", f"Wdq_{tag}"], [f"y_{tag}"], name=f"mm_{tag}"))
    outputs.append(helper.make_tensor_value_info(f"y_{tag}", TP.FLOAT, [1, N]))
    if with_vinfo:
        vinfo.append(helper.make_tensor_value_info(f"Wdq_{tag}", TP.FLOAT, [K, N]))

make_branch("a", TP.INT8, True)     # int8 quantized weight + value_info on DQ output
make_branch("b", TP.FLOAT, True)    # float32 int4-range weight + value_info
make_branch("c", TP.FLOAT, False)   # float32 no value_info (isolate the vinfo variable)

g = helper.make_graph(nodes, "g", inputs, outputs, inits, value_info=vinfo)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
m.ir_version = 8
onnx.save(m, "/home/kram/int4lab/test12.onnx")
print("saved test12")
