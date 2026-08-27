import sys

def parse(buf, off, end, depth, path, out, maxnodes=4000):
    while off < end and len(out) < maxnodes:
        start = off
        # varint tag
        tag = 0; shift = 0
        while True:
            b = buf[off]; off += 1
            tag |= (b & 0x7F) << shift; shift += 7
            if not (b & 0x80): break
        field, wt = tag >> 3, tag & 7
        if field == 0: return off
        if wt == 0:  # varint
            v = 0; shift = 0
            while True:
                b = buf[off]; off += 1
                v |= (b & 0x7F) << shift; shift += 7
                if not (b & 0x80): break
            if len(out) < 60: out.append(("  "*depth) + f"f{field} VARINT {v} @{start}")
        elif wt == 2:  # LEN
            ln = 0; shift = 0
            while True:
                b = buf[off]; off += 1
                ln |= (b & 0x7F) << shift; shift += 7
                if not (b & 0x80): break
            data_off = off; off += ln
            if ln > 200000 or (depth == 0 and ln > 1000):
                out.append(("  "*depth) + f"f{field} LEN {ln} @{start} data@{data_off} *** BIG")
                # peek inside big ones one level
                try: parse(buf, data_off, data_off+min(ln,200), depth+1, path+f".{field}", out, 60)
                except Exception: pass
            elif ln > 200:
                out.append(("  "*depth) + f"f{field} LEN {ln} @{start}")
                try: parse(buf, data_off, data_off+ln, depth+1, path+f".{field}", out, maxnodes)
                except Exception: pass
            else:
                txt = ''
                try: txt = buf[data_off:data_off+ln].decode('utf-8')[:40]
                except Exception: pass
                if len(out) < 200: out.append(("  "*depth) + f"f{field} LEN {ln} @{start} '{txt}'")
        elif wt == 5: off += 4
        elif wt == 1: off += 8
        else: return off

b = open('/tmp/int4lb2/out_s4/qwen3_p64_l0_together.axmodel','rb').read()
# f7 at offset 71 (from walk output), length from header
off = 71
tag = b[off]; off2 = off+1
ln = 0; shift = 0
while True:
    x = b[off2]; off2 += 1
    ln |= (x & 0x7F) << shift; shift += 7
    if not (x & 0x80): break
print(f"f7 header: tag={tag:#x} len={ln} data@{off2}")
out = []
parse(b, 71, 71+9503242, 0, "f7", out, 2500)
for line in out[:120]: print(line)
print("... total nodes:", len(out))
