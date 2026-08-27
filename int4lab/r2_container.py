import numpy as np, struct, sys

def load(p): return open(p,'rb').read()
b0 = load('/tmp/int4lb2/out_s4/qwen3_p64_l0_together.axmodel')
b1 = load('/tmp/int4lb2/out_s4/qwen3_p64_l1_together.axmodel')
print("l0:", len(b0), "l1:", len(b1))
a0 = np.frombuffer(b0, np.uint8); a1 = np.frombuffer(b1, np.uint8)
n = min(len(a0), len(a1))
diff = a0[:n] != a1[:n]
print("diff bytes: %d / %d (%.1f%%)" % (diff.sum(), n, 100*diff.mean()))
# contiguous diff regions
idx = np.flatnonzero(diff)
if len(idx):
    breaks = np.flatnonzero(np.diff(idx) > 64)
    starts = np.concatenate([[idx[0]], idx[breaks+1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]])
    regs = sorted(zip(starts, ends), key=lambda r: r[1]-r[0], reverse=True)[:12]
    print("top diff regions (start,end,len):")
    for s,e in regs: print("  %8d %8d %8d" % (s,e,e-s))
# nibble histogram of the biggest per-layer region (candidate weights)
s,e = regs[0]
reg = a0[s:e+1]
lo = reg & 0xF; hi = reg >> 4
hist = np.bincount(np.concatenate([lo, hi]), minlength=16)
print("nibble histogram biggest region:", hist.tolist())
print("entropy-ish: top2 share %.2f" % (np.sort(hist)[-2:].sum()/hist.sum()))
# f7 blob starts at 71 per walk; check protobuf-ish: tag byte at 71
print("bytes 71..90:", b0[71:91].hex())
