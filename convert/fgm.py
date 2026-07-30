"""fastgemma model format (.fgm) — writer/reader helpers.

Layout
------
    [0..8)    magic  b"FASTGEM1"
    [8..12)   u32 little-endian header length
    [12..)    header JSON (utf-8), then padding to 64B
    ...       tensor blobs, each 64B-aligned

Every tensor entry in the header carries {dtype, shape, offset, nbytes, meta}.
Quantised linear weights are stored *pre-packed for AMX*, so the runtime does
zero layout work at load time — it mmaps and executes.
"""

import json
import struct

MAGIC = b"FASTGEM1"
ALIGN = 64

# ---------------------------------------------------------------- AMX geometry
# _tile_dpbssd consumes A as 16 rows x 64 int8 (M=16, K=64) and B VNNI-packed as
# 16 rows x 64 bytes encoding (K=64, N=16) in [k/4][n][k%4] order.
TILE_M = 16
TILE_N = 16
TILE_K = 64


def pad_to(n, a=ALIGN):
    return (n + a - 1) // a * a


class FgmWriter:
    def __init__(self, path):
        self.f = open(path, "wb")
        self.tensors = {}
        self.meta = {}
        self.f.seek(0)
        self._cursor = 0
        self._body_started = False

    def start_body(self, header_reserve):
        """Reserve `header_reserve` bytes for the header, then write blobs after."""
        self._body_start = pad_to(12 + header_reserve)
        self._header_reserve = header_reserve
        self.f.seek(self._body_start)
        self._cursor = self._body_start
        self._body_started = True

    def add(self, name, arr, dtype, shape=None, meta=None):
        assert self._body_started, "call start_body() first"
        assert name not in self.tensors, f"duplicate tensor {name}"
        b = arr.tobytes() if hasattr(arr, "tobytes") else bytes(arr)
        off = pad_to(self._cursor)
        if off != self._cursor:
            self.f.write(b"\0" * (off - self._cursor))
        self.f.write(b)
        self.tensors[name] = {
            "dtype": dtype,
            "shape": list(shape if shape is not None else arr.shape),
            "offset": off,
            "nbytes": len(b),
            "meta": meta or {},
        }
        self._cursor = off + len(b)
        return self.tensors[name]

    # -- streaming: for tensors too large to materialise (PLE, lm_head) ------
    def begin(self, name, dtype, shape, meta=None):
        assert self._body_started, "call start_body() first"
        assert name not in self.tensors, f"duplicate tensor {name}"
        off = pad_to(self._cursor)
        if off != self._cursor:
            self.f.write(b"\0" * (off - self._cursor))
        self._cursor = off
        self._open = {
            "dtype": dtype,
            "shape": list(shape),
            "offset": off,
            "nbytes": 0,
            "meta": meta or {},
        }
        self._open_name = name
        return self._open

    def append(self, arr):
        b = arr.tobytes() if hasattr(arr, "tobytes") else bytes(arr)
        self.f.write(b)
        self._open["nbytes"] += len(b)
        self._cursor += len(b)

    def end(self):
        self.tensors[self._open_name] = self._open
        e, self._open, self._open_name = self._open, None, None
        return e

    def close(self, config):
        header = json.dumps(
            {"arch": "gemma3n", "config": config, "meta": self.meta, "tensors": self.tensors},
            separators=(",", ":"),
        ).encode()
        if len(header) > self._header_reserve:
            raise RuntimeError(
                f"header {len(header)}B exceeds reserve {self._header_reserve}B"
            )
        self.f.seek(0)
        self.f.write(MAGIC)
        self.f.write(struct.pack("<I", len(header)))
        self.f.write(header)
        self.f.close()


def read_header(path):
    with open(path, "rb") as f:
        assert f.read(8) == MAGIC, "not a .fgm file"
        (n,) = struct.unpack("<I", f.read(4))
        return json.loads(f.read(n))
