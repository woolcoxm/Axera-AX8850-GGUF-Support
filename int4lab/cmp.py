import onnx, numpy as np, sys
a = onnx.load(sys.argv[1]); b = onnx.load(sys.argv[2])
ta = {t.name: t for t in a.graph.initializer}; tb = {t.name: t for t in b.graph.initializer}
for n in ta:
    wa = np.frombuffer(ta[n].raw_data, dtype=np.float32)
    wb = np.frombuffer(tb[n].raw_data, dtype=np.float32) if n in tb else None
    if wb is None: print(n, "MISSING in output"); continue
    print(n, "identical:", np.array_equal(wa, wb), "| maxdiff:", float(np.abs(wa-wb).max()) if len(wa)==len(wb) else "len mismatch")
