"""Fingerprint the 11 llm_build2-s4 npu_params blobs: explain size variance."""
import numpy as np, os, sys

S = '/home/kram/Desktop/Projects/LLMTest/int4lab/scratch/claims'
BUILDS = ['mk0', 'mk1', 'mk_2', 'mk_3', 'mk_4', 'mk_5', 'mk_6', 'mk_7',
          'mk_d2', 'mk_mc', 'mk_mixamp']

blobs = {}
for b in BUILDS:
    p = os.path.join(S, f'{b}_s4_l0.npy')
    blobs[b] = np.load(p)

REF = blobs['mk0']

def entropy(a):
    h = np.bincount(a, minlength=256).astype(np.float64)
    h = h[h > 0] / a.size
    return float(-(h * np.log2(h)).sum())

print(f"{'build':10} {'size':>9}  {'H(byte)':>7} {'uniq':>4} "
      f"{'prefix_vs_mk0':>13} {'shared4k%':>9} {'zerofill%':>9} {'top3 bytes (val:count%)'}")
# 4KB block hashes for shared-region analysis
def blockhashes(a, bs=4096):
    n = a.size // bs
    return {a[i*bs:(i+1)*bs].tobytes() for i in range(n)}
ref_blocks = blockhashes(REF)
for b in BUILDS:
    a = blobs[b]
    d = a != REF[:a.size] if a.size <= REF.size else REF != a[:REF.size]
    # common prefix
    ne = np.flatnonzero(REF[:a.size] != a[:a.size]) if a.size <= REF.size else None
    prefix = int(ne[0]) if ne is not None and ne.size else min(a.size, REF.size)
    shared = len(blockhashes(a) & ref_blocks) / max(1, len(blockhashes(a)))
    h = np.bincount(a, minlength=256)
    top3 = sorted(enumerate(h), key=lambda t: -t[1])[:3]
    tot = a.size
    t3 = ' '.join(f'{v:02x}:{100*c/tot:.1f}' for v, c in top3)
    print(f"{b:10} {a.size:>9}  {entropy(a):>7.3f} {len(np.unique(a)):>4} "
          f"{prefix:>13} {100*shared:>8.1f}% {100*h[0]/tot:>8.1f}% {t3}")

print()
print("=== tail 32 bytes (hex) ===")
for b in BUILDS:
    print(f"{b:10}", blobs[b][-32:].tobytes().hex())

print()
print("=== head 64 bytes (hex) ===")
for b in BUILDS:
    print(f"{b:10}", blobs[b][:64].tobytes().hex())

# run-length stats on a low-entropy build (mk_mc) vs high (mk0)
def rle_stats(a):
    ch = a[1:] != a[:-1]
    brk = np.flatnonzero(ch)
    bounds = np.concatenate(([-1], brk, [a.size-1]))
    return np.diff(bounds)

for b in ['mk_mc', 'mk_2', 'mk_3', 'mk0', 'mk_mixamp']:
    a = blobs[b]
    ch = a[1:] != a[:-1]
    brk = np.flatnonzero(ch)
    bounds = np.concatenate(([-1], brk, [a.size-1]))
    runs = np.diff(bounds)
    print(f"{b:10} runs={runs.size:>8} max={runs.max():>7} mean={runs.mean():>7.2f} "
          f"runs>=16={int((runs>=16).sum()):>7} bytes_in_runs>=16={int(runs[runs>=16].sum()):>9}")
