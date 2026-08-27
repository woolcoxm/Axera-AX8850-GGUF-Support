import os, numpy as np
S = '/home/kram/Desktop/Projects/LLMTest/int4lab/scratch'
def npu_params(buf):
    i = 0
    while True:
        i = buf.find(b'\x42\x0anpu_params\x4a', i+1)
        if i < 0: return None
        p = i + 12; ln = 0; sh = 0; q = p+1
        while True:
            b = buf[q]; q += 1; ln |= (b & 0x7F) << sh; sh += 7
            if not (b & 0x80): break
        if ln > 100000: return buf[q:q+ln]
os.makedirs(os.path.join(S, 'claims2'), exist_ok=True)
for d in sorted(os.listdir(S)):
    if not d.endswith('_s4'): continue
    for L in (0, 1):
        src = os.path.join(S, d, f'qwen3_l{L}.axmodel')
        if not os.path.exists(src): continue
        np_ = npu_params(open(src, 'rb').read())
        assert np_ is not None, (d, L)
        np.save(os.path.join(S, 'claims2', f'{d}_l{L}.npy'), np.frombuffer(np_, np.uint8))
        print(d, f'l{L}', len(np_), flush=True)
print('done')
