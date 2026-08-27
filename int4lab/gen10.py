import numpy as np, onnx
from onnx import helper, TensorProto as TP
from onnx import external_data_helper as edh
K, N = 256, 512
k = np.arange(K)[:, None]; n = np.arange(N)[None, :]
W = ((3*k + 5*n) % 16 - 8).astype(np.float32)
X = helper.make_tensor_value_info("X", TP.FLOAT, [1, K])
Y = helper.make_tensor_value_info("Y", TP.FLOAT, [1, N])
node = helper.make_node("MatMul", ["X", "W"], ["Y"], name="mm0")
Wt = helper.make_tensor("W", TP.FLOAT, [K, N], W.tobytes(), raw=True)
edh.set_external_data(Wt, location="no_such_dir/weights.bin")
Wt.raw_data = b""
Wt.data_location = TP.EXTERNAL
g = helper.make_graph([node], "g", [X], [Y], [Wt])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
m.ir_version = 8
open("/home/kram/int4lab/test10.onnx","wb").write(m.SerializeToString())
print("saved test10 (external data -> missing file)")
