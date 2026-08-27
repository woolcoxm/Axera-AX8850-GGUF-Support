import onnx, sys
from onnx import TensorProto
m = onnx.load(sys.argv[1])
print("IR version:", m.ir_version)
for oi in m.opset_import:
    print("opset:", oi.domain or "(default)", oi.version)
g = m.graph
print("\n== INPUTS ==");  [print(f"  {i.name}: {TensorProto.DataType.Name(i.type.tensor_type.elem_type)} {[(d.dim_value or d.dim_param) for d in i.type.tensor_type.shape.dim]}") for i in g.input]
print("== OUTPUTS =="); [print(f"  {o.name}: {TensorProto.DataType.Name(o.type.tensor_type.elem_type)} {[(d.dim_value or d.dim_param) for d in o.type.tensor_type.shape.dim]}") for o in g.output]
print("== NODES ==")
for nd in g.node:
    print(f"  {nd.name}: op={nd.op_type} domain='{nd.domain}' inputs={list(nd.input)} outputs={list(nd.output)}")
    for a in nd.attribute:
        av = onnx.helper.get_attribute_value(a)
        if isinstance(av, bytes):
            try: av = av.decode()
            except: av = f"<{len(av)} bytes>"
        print(f"      attr {a.name} = {av}")
print("== INITIALIZERS ==")
TP = TensorProto.DataType
for t in g.initializer:
    nbytes = len(t.raw_data) if t.raw_data else 0
    dims = list(t.dims)
    print(f"  {t.name}: dtype={TP.Name(t.data_type)} dims={dims} raw={nbytes}B" + (f" (external: {t.data_location})" if t.data_location else ""))
