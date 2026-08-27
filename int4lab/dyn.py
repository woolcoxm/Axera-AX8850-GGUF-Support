import onnx, sys
m = onnx.load("/home/kram/int4lab/conv1x1_int4.onnx")
from onnx import TensorProto as TP
t = [x for x in m.graph.initializer if x.name == "Wq"][0]
m.graph.initializer.remove(t)
inp = m.graph.input.add()
inp.name = "Wq"
inp.type.tensor_type.elem_type = t.data_type   # INT4 as INPUT
for d in t.dims: inp.type.tensor_type.shape.dim.add().dim_value = d
onnx.save(m, "/home/kram/int4lab/conv1x1_int4_dyn.onnx")
print("Wq moved to graph input (INT4-typed)")
