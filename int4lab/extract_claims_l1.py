import os
import numpy as np
S = os.environ['S']
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
os.makedirs(os.path.join(S, 'claims'), exist_ok=True)
for d in sorted(os.listdir(S)):
    if not d.endswith('_s4'): continue
    src = os.path.join(S, d, 'qwen3_l1.axmodel')
    if not os.path.exists(src): continue
    np_ = npu_params(open(src, 'rb').read())
    if np_ is None:
        print(d, 'NO npu_params'); continue
    np.save(os.path.join(S, 'claims', f'{d}_l0.npy'), np.frombuffer(np_, np.uint8))
    print(d, len(np_), flush=True)
print('extraction done')
