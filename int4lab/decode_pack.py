import onnx, numpy as np, sys
m = onnx.load(sys.argv[1])
ts = {t.name: t for t in m.graph.initializer}
t = ts.get("Wq_c13")
raw = np.frombuffer(t.raw_data, dtype=np.uint8)
print("packed bytes:", len(raw), "first 16:", raw[:16].tolist())
# known codes: wc = ((idx*7+3) % 16 - 8) over (8,4,3,3) C-order
idx = np.arange(8*4*3*3)
codes = ((idx*7+3) % 16 - 8).astype(np.int8).reshape(8,4,3,3)
flat = codes.reshape(-1)
# unpack hypothesis A: ORT pack_bytes_to_4bit = low nibble first, val = code (signed?) stored as (code & 0xF)?
# try: nibble pairs (lo=elem0, hi=elem1), value = lo (unsigned 0..15)
lo = (raw & 0xF).astype(np.int8); hi = (raw >> 4).astype(np.int8)
# candidate mappings of unsigned nibble -> code
def signed(x): # interpret nibble as 4-bit two's complement
    return np.where(x >= 8, x - 16, x)
def offset(x): # code + 8
    return (x - 8).astype(np.int8)
for name, dec in (("twos", signed), ("offset8", offset), ("raw", lambda x: x)):
    seq = np.empty(len(raw)*2, np.int8)
    seq[0::2] = dec(lo); seq[1::2] = dec(hi)
    match = int((seq[:len(flat)] == flat).sum())
    print(f"{name}: {match}/{len(flat)} match")
# reverse order variant
for name, dec in (("twos_hi_first", signed), ("offset_hi_first", offset)):
    seq = np.empty(len(raw)*2, np.int8)
    seq[0::2] = dec(hi); seq[1::2] = dec(lo)
    match = int((seq[:len(flat)] == flat).sum())
    print(f"{name}: {match}/{len(flat)} match")
