import types, sys
import convert_onnx_to_4w8f as M

def dump_code(c, depth=0, seen=None):
    seen = seen if seen is not None else set()
    if c in seen or depth > 5: return
    seen.add(c)
    print("=" * 60)
    print(f"{c.co_firstlineno} def {c.co_name} args={c.co_varnames[:c.co_argcount]}")
    print("  names:", c.co_names)
    consts = []
    for k in c.co_consts:
        if isinstance(k, types.CodeType): consts.append(f"<code {k.co_name}@{k.co_firstlineno}>")
        else: consts.append(repr(k)[:160])
    print("  consts:", consts)
    for k in c.co_consts:
        if isinstance(k, types.CodeType): dump_code(k, depth+1, seen)

for name in ("is_int4_range", "is_weight_4bits", "constant_to_onnx_tensor"):
    f = getattr(M, name, None)
    if f: dump_code(f.__code__)
Ax = getattr(M, "AxOnnxExporter", None)
if Ax:
    for mname, m in sorted(vars(Ax).items()):
        if callable(m) and hasattr(m, "__code__"):
            print("CLASS METHOD:", mname); dump_code(m.__code__)
print("INT4_DTYPE =", getattr(M, "INT4_DTYPE", None))
