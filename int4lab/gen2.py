import numpy as np, onnx
from onnx import helper, TensorProto as TP

K = N = 64
rng = np.random.default_rng(7)
def w(name, shape):
    vals = rng.standard_normal(shape).astype(np.float32)
    t = helper.make_tensor(name, TP.FLOAT, shape, vals.tobytes(), raw=True)
    return vals, t

nodes, inits, inputs, outputs = [], [], [], []
inputs.append(helper.make_tensor_value_info("X", TP.FLOAT, [1, K]))
inputs.append(helper.make_tensor_value_info("Xm", TP.FLOAT, [8, K]))
inputs.append(helper.make_tensor_value_info("Xi", TP.FLOAT, [1, 8, 8, 8]))

# b0 plain MatMul W[K,N]
v0, t0 = w("W0", [K, N]); inits.append(t0)
nodes.append(helper.make_node("MatMul", ["X","W0"], ["Y0"], name="b0"))

# b1 Transpose(W[N,K])
v1, t1 = w("W1", [N, K]); inits.append(t1)
nodes.append(helper.make_node("Transpose", ["W1"], ["T1"], perm=[1,0], name="tr1"))
nodes.append(helper.make_node("MatMul", ["X","T1"], ["Y1"], name="b1"))

# b2 Gemm transB
v2, t2 = w("W2", [N, K]); inits.append(t2)
bv = np.zeros(N, np.float32)
inits.append(helper.make_tensor("B2", TP.FLOAT, [N], bv.tobytes(), raw=True))
nodes.append(helper.make_node("Gemm", ["X","W2","B2"], ["Y2"], transB=1, name="b2"))

# b3 transformer-style names
v3, t3 = w("model.layers.0.self_attn.q_proj.weight", [K, N]); inits.append(t3)
nodes.append(helper.make_node("MatMul", ["X","model.layers.0.self_attn.q_proj.weight"], ["Y3"], name="model.layers.0.self_attn.q_proj"))

# b5 weight-side QDQ: MatMul(X, DequantizeLinear(W5q,s5,z5))
q5 = rng.integers(-127, 128, (K,N), dtype=np.int8); s5 = np.float32([0.02])
inits.append(helper.make_tensor("W5q", TP.INT8, [K,N], q5.tobytes(), raw=True))
inits.append(helper.make_tensor("s5", TP.FLOAT, [], s5.tobytes(), raw=True))
inits.append(helper.make_tensor("z5", TP.INT8, [], np.int8(0).tobytes(), raw=True))
nodes.append(helper.make_node("DequantizeLinear", ["W5q","s5","z5"], ["W5d"], name="dq5"))
nodes.append(helper.make_node("MatMul", ["X","W5d"], ["Y5"], name="b5"))

# b6 Conv 1x1
vc, tc = w("Wc", [16, 8, 1, 1]); inits.append(tc)
nodes.append(helper.make_node("Conv", ["Xi","Wc"], ["Y6"], name="b6"))

# b7 batched MatMul
v7, t7 = w("W7", [K, N]); inits.append(t7)
nodes.append(helper.make_node("MatMul", ["Xm","W7"], ["Y7"], name="b7"))

shapes = {"Y0":[1,N],"Y1":[1,N],"Y2":[1,N],"Y3":[1,N],"Y5":[1,N],"Y6":[1,16,8,8],"Y7":[8,N]}
for nm, sh in shapes.items():
    outputs.append(helper.make_tensor_value_info(nm, TP.FLOAT, sh))

g = helper.make_graph(nodes, "g", inputs, outputs, inits)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
m.ir_version = 8
onnx.checker.check_model(m)
onnx.save(m, "/home/kram/int4lab/test2.onnx")
print("saved test2.onnx")
