#!/usr/bin/env python3
"""Extract the npu_params blob from a (vendor w8a16) layer axmodel.

The blob is the big f5 inside f7 whose inner name field is "npu_params".
Writes <out> and prints its offset/length. Verifies the byte after the
blob is the start of the next protobuf field.
"""
import sys


def read_varint(buf, p):
    v = 0
    shift = 0
    while True:
        b = buf[p]
        p += 1
        v |= (b & 0x7F) << shift
        if not (b & 0x80):
            return v, p
        shift += 7


def extract(path, out):
    buf = open(path, 'rb').read()
    # top-level f7
    tag, p = read_varint(buf, 0)  # f1 varint
    assert tag == 8, 'expected f1 varint'
    _, p = read_varint(buf, p)
    tag, p = read_varint(buf, p)  # f2 "Pulsar2"
    ln, p = read_varint(buf, p); p += ln
    tag, p = read_varint(buf, p)  # f6 version
    ln, p = read_varint(buf, p); p += ln
    tag, p = read_varint(buf, p)  # f7
    assert tag >> 3 == 7 and tag & 7 == 2, f'unexpected tag {tag:#x}'
    ln, p = read_varint(buf, p)
    f7 = buf[p:p + ln]
    # inside f7: walk fields until the big f5
    q = 0
    while q < len(f7):
        tag, q2 = read_varint(f7, q)
        wt = tag & 7
        if wt == 2:
            ln2, q2 = read_varint(f7, q2)
            if tag >> 3 == 5 and ln2 > 10_000_000:
                payload = f7[q2:q2 + ln2]
                # inner: f1 varint (data len), f2 varint, f8 "npu_params", data
                r = 0
                while r < len(payload):
                    it, r2 = read_varint(payload, r)
                    if it & 7 == 2:
                        ln, r3 = read_varint(payload, r2)
                        if payload[r3:r3 + ln] == b'npu_params':
                            # next field is f9 = the data itself
                            it2, r4 = read_varint(payload, r3 + ln)
                            assert it2 >> 3 == 9 and it2 & 7 == 2, f'unexpected data tag {it2:#x}'
                            ln3, r5 = read_varint(payload, r4)
                            blob = payload[r5:r5 + ln3]
                            open(out, 'wb').write(blob)
                            print(f'{path}: npu_params abs_off={p + q2 + r5} len={ln3} '
                                  f'(f5 payload ends at {p + q2 + ln2}, trailing {ln2 - r5 - ln3}B)')
                            return ln3
                        r = r3 + ln
                    elif it & 7 == 0:
                        _, r = read_varint(payload, r2)
                    elif it & 7 == 5:
                        r = r2 + 4
                    elif it & 7 == 1:
                        r = r2 + 8
                    else:
                        break
            q = q2 + ln2
        elif wt == 0:
            _, q = read_varint(f7, q2)
        elif wt == 1:
            q = q2 + 8
        else:
            q = q2 + 4
    raise ValueError('npu_params not found')


if __name__ == '__main__':
    extract(sys.argv[1], sys.argv[2])
