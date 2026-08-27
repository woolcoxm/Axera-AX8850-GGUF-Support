import numpy as np, onnx
from onnx import helper, TensorProto as TP
K = N = 64
k = np.arange(K)[:, None]; n = np.arange(N)[None, :]
codes = ((3*k + 5*n) % 16 - 8).astype(np.int8)          # int8 dtype, int4-range values
Wt8 = helper.make_tensor("W8", TP.INT8, [K, N], codes.tobytes(), raw=True)
full = ((codes.astype(np.int32) * 17) % 255 - 127).astype(np.int8)  # full int8 range control
WtF = helper.make_tensor("WF", TP.INT8, [K, N], full.tobytes(), raw=True)
nodes, outputs = [], []
X = helper.make_tensor_value_info("X", TP.FLOAT, [1, K])
nodes.append(helper.make_node("MatMul", ["X", "W8"], ["y0"], name="n0_mm_int4range"))
nodes.append(helper.make_node("Gemm", ["X", "W8"], ["y1"], name="n1_gemm_int4range", transB=1))
nodes.append(helper.make_node("MatMul", ["X", "WF"], ["y2"], name="n2_mm_fullrange"))
nodes.append(helper.make_node("AxQuantizedMatMul", ["X", "W8"], ["y3"], name="n3_axq_int4range",
    input_scales=(1.0, 1.0), input_zeropoints=(0, 0), output_dtype="FP32",
    output_scales=(1.0,), output_zeropoints=(0,), quant_method=0, transposed=1, domain="ax.matmul_gen"))
for i in range(4):
    outputs.append(helper.make_tensor_value_info(f"y{i}", TP.FLOAT, [1, N]))
g = helper.make_graph(nodes, "g", [X], outputs, [Wt8, WtF])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("ax.matmul_gen", 1)])
m.ir_version = 8
onnx.save(m, "/home/kram/int4lab/test9.onnx")
print("saved test9")
