import sys, types
DUMP = open('/home/kram/int4lab/code_dump6.txt', 'w')

def dump_code(c, tag):
    DUMP.write("\n" + "#"*70 + f"\n### {tag} @ {c.co_firstlineno} def {c.co_name} args={c.co_varnames[:c.co_argcount]}\n")
    try:
        s = _io.StringIO(); _dis.dis(c, file=s, depth=0)
        DUMP2.write("\n" + "="*70 + f"\n### DISASM {tag} @ {c.co_firstlineno} {c.co_name}\n" + s.getvalue() + "\n")
    except Exception as e:
        DUMP2.write(f"dis err {tag}: {e}\n")
    try: DUMP.write("names: " + repr(c.co_names) + "\n")
    except Exception as e: DUMP.write(f"names err {e}\n")
    try:
        consts = []
        for k in c.co_consts:
            if isinstance(k, types.CodeType): consts.append(f"<code {k.co_name}@{k.co_firstlineno}>")
            else: consts.append(repr(k)[:250])
        DUMP.write("consts: " + "; ".join(consts) + "\n")
    except Exception as e: DUMP.write(f"consts err {e}\n")

def hook(t, v, tb):
    DUMP.write(f"====== EXCEPT {t.__name__}: {v} ======\n")
    frames = []
    n = 0
    while tb is not None and n < 40:
        frames.append(tb.tb_frame); tb = tb.tb_next; n += 1
    # harvest module functions from armored module frame locals
    for fr in frames:
        fn = fr.f_code.co_filename
        if 'convert_onnx_to_4w8f' in fn and fr.f_code.co_name == '<module>':
            loc = fr.f_locals
            for fname in ("is_int4_range", "is_weight_4bits", "constant_to_onnx_tensor", "export_onnx"):
                f = loc.get(fname)
                if f is not None and hasattr(f, '__code__'):
                    try: dump_code(f.__code__, "HARVEST " + fname)
                    except Exception as e: DUMP.write(f"{fname} harvest err {e}\n")
            Ax = loc.get("AxOnnxExporter")
            if Ax is not None:
                try:
                    for mname, m in sorted(vars(Ax).items()):
                        if callable(m) and hasattr(m, '__code__'):
                            dump_code(m.__code__, "HARVEST AxOnnxExporter." + mname)
                except Exception as e: DUMP.write(f"Ax harvest err {e}\n")
            break
    DUMP.flush()
    sys.__excepthook__(t, v, tb)

sys.excepthook = hook
DUMP.write("harvesting excepthook v10 armed\n")
import dis as _dis, io as _io
DUMP2 = open('/home/kram/int4lab/disasm.txt', 'w')
