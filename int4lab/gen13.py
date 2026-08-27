import numpy as np, onnx
from onnx import helper, TensorProto as TP
K = N = 32
k = np.arange(K)[:, None]; n = np.arange(N)[None, :]
codes = ((3*k + 5*n) % 16 - 8)
nodes, inits, inputs, outputs, vinfo = [], [], [], [], []
inputs.append(helper.make_tensor_value_info("X", TP.FLOAT, [1, K]))
inputs.append(helper.make_tensor_value_info("Xi", TP.FLOAT, [1, 4, 8, 8]))
cnt = [0]
def dq_branch(warr, wdtype, consumer, per_channel=False, tag=None):
    tag = tag or f"v{cnt[0]}"; cnt[0] += 1
    inits.append(helper.make_tensor(f"Wq_{tag}", wdtype, list(warr.shape), warr.tobytes(), raw=True))
    sshape = [N] if per_channel else []
    svals = np.full(sshape, 0.5, np.float32)
    inits.append(helper.make_tensor(f"Ws_{tag}", TP.FLOAT, sshape, svals.tobytes(), raw=True))
    inits.append(helper.make_tensor(f"Wz_{tag}", TP.INT8, [] if not per_channel else [N], np.zeros(N if per_channel else 1, np.int8).tobytes(), raw=True))
    nodes.append(helper.make_node("DequantizeLinear", [f"Wq_{tag}", f"Ws_{tag}", f"Wz_{tag}"], [f"Wdq_{tag}"], name=f"dq_{tag}"))
    vinfo.append(helper.make_tensor_value_info(f"Wdq_{tag}", TP.FLOAT, list(warr.shape)))
    if consumer == "mm":
        nodes.append(helper.make_node("MatMul", ["X", f"Wdq_{tag}"], [f"y_{tag}"], name=f"use_{tag}"))
        outputs.append(helper.make_tensor_value_info(f"y_{tag}", TP.FLOAT, [1, warr.shape[1]]))
    else:
        nodes.append(helper.make_node("Conv", ["Xi", f"Wdq_{tag}"], [f"y_{tag}"], name=f"use_{tag}"))
        outputs.append(helper.make_tensor_value_info(f"y_{tag}", TP.FLOAT, [1, warr.shape[0], 6, 6]))

wc4 = ((np.arange(8*4*3*3).reshape(8,4,3,3)*7+3) % 16 - 8)
dq_branch(codes.astype(np.int8), TP.INT8, "mm", tag="a13")              # int8 codes, MatMul
dq_branch(codes.astype(np.float32), TP.FLOAT, "mm", tag="b13")          # float codes, MatMul
dq_branch(wc4.astype(np.int8), TP.INT8, "conv", tag="c13")              # int8 4D conv weight via DQ
dq_branch(codes.astype(np.int8), TP.INT8, "mm", per_channel=True, tag="d13")  # per-channel scale
g = helper.make_graph(nodes, "g", inputs, outputs, inits, value_info=vinfo)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
m.ir_version = 8
onnx.save(m, "/home/kram/int4lab/test13.onnx")
print("saved test13 (opset 13)")
