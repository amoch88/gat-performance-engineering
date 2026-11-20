"""Canonical EIDs with explicit sparse storage; no coalescing or node reorder."""
from dataclasses import dataclass
import numpy as np

FORMATS = [(name, 0) for name in ("coo", "csr", "csc")] + [
    (name, size) for name in ("bcsr", "bcsc", "csb") for size in (16, 32, 64)
]
KINDS = ("coo", "csr", "csc", "bcsr", "bcsc", "csb")


def packed(x):
    return np.ascontiguousarray(x, dtype=np.int32)


def ptr(ids, count):
    return packed(np.r_[0, np.cumsum(np.bincount(ids, minlength=count))])


@dataclass(frozen=True)
class Storage:
    n: int
    e: int
    name: str
    block: int
    tiles: int
    arrays: tuple

    @property
    def bytes(self):
        return sum(a.nbytes for a in self.arrays)

    def validate(self, src, dst):
        """Decode the stored coordinates independently of the native traversal."""
        a, b, c, d, f, ids = self.arrays
        if self.name == "coo":
            u, v, ids = a, b, np.arange(self.e)
        elif self.name in ("csr", "csc"):
            row = np.repeat(np.arange(self.n), np.diff(a))
            u, v = (row, b) if self.name == "csr" else (b, row)
        elif self.name == "csb":
            u = np.repeat(b, np.diff(a)) * self.block + d
            v = np.repeat(c, np.diff(a)) * self.block + f
        else:
            major = np.repeat(np.arange(len(a) - 1), np.diff(a))
            sizes = np.diff(d)
            assert len(sizes) == self.tiles * self.block
            row = np.repeat(major.repeat(self.block) * self.block
                            + np.tile(np.arange(self.block), self.tiles), sizes)
            col = np.repeat(b, sizes.reshape(-1, self.block).sum(1)) * self.block + f
            u, v = (row, col) if self.name == "bcsr" else (col, row)
        np.testing.assert_array_equal(np.sort(ids), np.arange(self.e))
        np.testing.assert_array_equal(u, src[ids])
        np.testing.assert_array_equal(v, dst[ids])


def build(n, src, dst, name, block):
    if (name, block) not in FORMATS:
        raise ValueError("unknown sparse format/block size")
    e = len(src)
    if (not 0 <= n < 2**31 or e >= 2**31 or src.shape != (e,) or dst.shape != (e,)
            or src.dtype.kind not in "iu" or dst.dtype.kind not in "iu"
            or any(np.any((a < 0) | (a >= n)) for a in (src, dst))):
        raise ValueError("invalid graph or int32 storage overflow")
    empty = packed([])
    tiles = 0
    if name == "coo":
        arrays = (packed(src), packed(dst), empty, empty, empty, empty)
    elif name in ("csr", "csc"):
        major, minor = (src, dst) if name == "csr" else (dst, src)
        ids = np.argsort(major, kind="stable")
        arrays = (ptr(major, n), packed(minor[ids]), empty, empty, empty, packed(ids))
    else:
        major, minor = (dst, src) if name == "bcsc" else (src, dst)
        mb, nb = major // block, minor // block
        local_major, local_minor = major % block, minor % block
        ids = np.lexsort((local_major, nb, mb))
        mb, nb = mb[ids], nb[ids]
        local_major, local_minor = local_major[ids], local_minor[ids]
        starts = np.flatnonzero(np.r_[True, (mb[1:] != mb[:-1]) | (nb[1:] != nb[:-1])]) if e else np.array([], dtype=np.int64)
        tiles = len(starts)
        tile_ptr = np.r_[starts, e]
        if name == "csb":
            arrays = (packed(tile_ptr), packed(mb[starts]), packed(nb[starts]),
                      packed(local_major), packed(local_minor), packed(ids))
        else:
            if tiles * block >= 2**31:
                raise ValueError("tile pointer table too large")
            tile_ids = np.repeat(np.arange(tiles), np.diff(tile_ptr))
            arrays = (ptr(mb[starts], (n + block - 1) // block), packed(nb[starts]), empty,
                      ptr(tile_ids * block + local_major, tiles * block),
                      packed(local_minor), packed(ids))
    return Storage(n, e, name, block, tiles, arrays)
