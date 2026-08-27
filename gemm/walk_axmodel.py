#!/usr/bin/env python3
"""Walk an axmodel's protobuf wire format and dump the field tree.

Prints tag / wiretype / offset / length for every field, recursing into
sub-messages when the payload parses cleanly. Raw blobs >= 4KB are called
out as blob candidates (npu_params, microcode segments).
"""
import sys
import struct


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
        if shift > 63:
            raise ValueError


def try_message(buf, depth, path, out, max_depth=8):
    """Return True if buf parses fully as a sequence of valid fields."""
    p = 0
    n = len(buf)
    fields = []
    while p < n:
        try:
            tag, p2 = read_varint(buf, p)
        except (IndexError, ValueError):
            return False
        wt = tag & 7
        fn = tag >> 3
        if fn == 0 or wt not in (0, 1, 2, 5):
            return False
        if wt == 0:
            try:
                _, p2 = read_varint(buf, p2)
            except (IndexError, ValueError):
                return False
        elif wt == 1:
            p2 += 8
        elif wt == 5:
            p2 += 4
        else:
            try:
                ln, p2 = read_varint(buf, p2)
            except (IndexError, ValueError):
                return False
            if ln > n - p2:
                return False
            p2 += ln
        if p2 > n:
            return False
        fields.append((p, fn, wt, p2))
        p = p2
    return fields if fields else False


def looks_text(b):
    if not b:
        return False
    try:
        s = b.decode('utf-8')
    except UnicodeDecodeError:
        return False
    return all(c.isprintable() or c in '\n\t' for c in s)


def walk(buf, base, depth, path, out, max_depth=8):
    if depth > max_depth:
        return
    fields = try_message(buf, depth, path, out, max_depth)
    if not fields:
        return
    for (off, fn, wt, end) in fields:
        here = base + off
        if wt == 2:
            tag, p2 = read_varint(buf, off)
            ln, p2 = read_varint(buf, p2)
            payload = buf[p2:p2 + ln]
            abs_off = base + p2
            sub = None
            if ln > 0 and ln < 1 << 20 and depth < max_depth:
                sub = try_message(payload, depth + 1, path + [fn], out, max_depth)
            label = ''
            if ln <= 64 and looks_text(payload):
                label = ' text=' + repr(payload.decode('utf-8'))
            elif ln < 8:
                label = ' hex=' + payload.hex()
            out.append(f'{"  " * depth}f{fn} LEN {ln} @ {abs_off}{label}')
            if ln >= 4096:
                out.append(f'{"  " * depth}  ^ BLOB {ln} bytes @ {abs_off}')
            if sub:
                walk(payload, abs_off, depth + 1, path + [fn], out, max_depth)
            elif sub is None and ln >= 4096:
                pass  # blob, already noted
        else:
            tag, p2 = read_varint(buf, off)
            if wt == 0:
                v, _ = read_varint(buf, p2)
                out.append(f'{"  " * depth}f{fn} VARINT {v} @ {here}')
            elif wt == 1:
                out.append(f'{"  " * depth}f{fn} I64 {struct.unpack("<Q", buf[p2:p2+8])[0]} @ {here}')
            elif wt == 5:
                out.append(f'{"  " * depth}f{fn} I32 {struct.unpack("<I", buf[p2:p2+4])[0]} @ {here}')


def main():
    path = sys.argv[1]
    maxdepth = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    buf = open(path, 'rb').read()
    out = [f'== {path} ({len(buf)} bytes) ==']
    walk(buf, 0, 0, [], out, maxdepth)
    print('\n'.join(out))


if __name__ == '__main__':
    main()
