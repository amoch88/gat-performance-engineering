"""Dense adjacency tiles and destination-row sliced ELLPACK."""
import ctypes
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import subprocess

import numpy as np

from experiment import FLAGS, OPERATIONS
from common import H, F, torch
from storage import packed, ptr

HERE = Path(__file__).resolve().parent


class StorageLimit(ValueError):
    def __init__(self, required, limit):
        self.required, self.limit = required, limit
        super().__init__(f"Index payload requires {required:,} bytes, exceeding the {limit:,}-byte budget")


@dataclass(frozen=True)
class Padded:
    kind: str
    n: int
    e: int
    size: int
    groups: int
    slots: int
    arrays: tuple

    @property
    def bytes(self):
        return sum(a.nbytes for a in self.arrays)

    def validate(self, src, dst):
        a,b,c,d,extra = self.arrays
        if self.kind == "dense":
            groups, rows, cols = self.occupied(c)
            eid = d[groups, rows, cols]
            np.testing.assert_array_equal(a[groups]*self.size+rows, src[eid])
            np.testing.assert_array_equal(b[groups]*self.size+cols, dst[eid])
            # Check that duplicate-sidecar edges really belong to occupied cells.
            occupied = src[eid].astype(np.int64)*self.n + dst[eid]
            extra_src, extra_dst, extra_eid = extra.T
            np.testing.assert_array_equal(extra_src, src[extra_eid])
            np.testing.assert_array_equal(extra_dst, dst[extra_eid])
            assert np.isin(extra_src.astype(np.int64)*self.n+extra_dst, occupied).all()
            eid = np.r_[eid, extra_eid]
        else:
            positions = np.flatnonzero(c >= 0)
            groups = np.searchsorted(a,positions,side="right")-1
            rows = groups*self.size+(positions-a[groups])%self.size
            eid = c[positions]
            np.testing.assert_array_equal(b[positions],src[eid])
            np.testing.assert_array_equal(rows,dst[eid])
            assert np.all(b[c < 0] == -1)
        np.testing.assert_array_equal(np.sort(eid), np.arange(self.e))

    @staticmethod
    def occupied(masks):
        """Decode set bits in O(edges + rows), with bounded temporary memory."""
        tiles, rows = np.nonzero(masks)
        words = masks[tiles,rows]
        tile_chunks, row_chunks, col_chunks = [], [], []
        for col in range(masks.shape[1]):
            valid = (words >> np.uint64(col)) & np.uint64(1) != 0
            tile_chunks.append(tiles[valid])
            row_chunks.append(rows[valid])
            col_chunks.append(np.full(np.count_nonzero(valid),col,dtype=np.int64))
        return tuple(np.concatenate(chunks) for chunks in (tile_chunks,row_chunks,col_chunks))


def build_padded(n, src, dst, kind, size, limit):
    if kind not in ("dense","sell") or size not in (2,4,8,16,32,64):
        raise ValueError("unknown padded format or size")
    e = len(src)
    empty = packed([])
    if kind == "dense":
        u, v = src//size, dst//size
        row, col = src%size, dst%size
        order = np.lexsort((col,row,v,u))
        u,v,row,col = (values[order] for values in (u,v,row,col))
        tile_change = np.r_[True,(u[1:]!=u[:-1]) | (v[1:]!=v[:-1])][:e]
        cell_change = tile_change | np.r_[True,(row[1:]!=row[:-1]) | (col[1:]!=col[:-1])][:e]
        groups = int(tile_change.sum())
        slots = groups*size*size
        # One dense int32 EID per slot, one uint64 occupancy word per tile row.
        # Only parallel edges after the first in each cell require a sidecar.
        extra_count = e-int(cell_change.sum())
        required = 4*slots + 8*groups*(size+1) + 12*extra_count
        if required > limit:
            raise StorageLimit(required,limit)
        group = (np.cumsum(tile_change)-1)[cell_change]
        rows,cols = row[cell_change],col[cell_change]
        masks = np.zeros((groups,size),dtype=np.uint64)
        np.bitwise_or.at(masks,(group,rows),np.left_shift(np.uint64(1),cols.astype(np.uint64)))
        # Padding is initialized; the occupancy mask gates all reads.
        dense = np.full((groups,size,size),-1,dtype=np.int32)
        dense[group,rows,cols] = order[cell_change]
        extra = order[~cell_change]
        sidecar = packed(np.stack((src[extra],dst[extra],extra),axis=1))
        arrays = (packed(u[tile_change]),packed(v[tile_change]),masks,dense,sidecar)
    else:
        groups = (n+size-1)//size
        degree = np.bincount(dst,minlength=n)
        padded = np.pad(degree,(0,groups*size-n))
        width = padded.reshape(groups,size).max(axis=1)
        offsets = np.r_[0,np.cumsum(width*size)]
        slots = int(offsets[-1])
        required = 4*(groups+1) + 8*slots
        if required > limit or slots >= 2**31:
            raise StorageLimit(required,min(limit,4*(groups+1)+8*(2**31-1)))
        eid = np.argsort(dst,kind="stable")
        rows = dst[eid]
        ordinal = np.arange(e)-ptr(dst,n)[rows]
        positions = offsets[rows//size] + ordinal*size + rows%size
        sources = np.full(slots,-1,dtype=np.int32)
        edges = np.full_like(sources,-1)
        sources[positions],edges[positions] = src[eid],eid
        arrays = (packed(offsets),sources,edges,empty,empty)
    result = Padded(kind,n,e,size,groups,slots,arrays)
    assert result.bytes == required
    return result


def library():
    compiler = os.environ.get("CXX","g++")
    version = subprocess.check_output([compiler,"--version"],text=True).splitlines()[0]
    source = HERE/"padded.cpp"
    key = hashlib.sha256(source.read_bytes()+repr((version,FLAGS)).encode()).hexdigest()
    directory = Path("results/comprehensive_sparse_build")
    directory.mkdir(parents=True,exist_ok=True)
    target = directory/f"{key}.so"
    if not target.exists():
        temp = target.with_suffix(f".{os.getpid()}.tmp")
        subprocess.run([compiler,*FLAGS,str(source),"-o",str(temp)],check=True)
        temp.replace(target)
    binary = ctypes.CDLL(str(target.resolve()))
    binary.padded_run.argtypes = ([ctypes.c_int]*6+[ctypes.c_void_p]*7+
                                  [ctypes.c_float]+[ctypes.c_void_p]*3)
    binary.padded_run.restype = None
    return binary, {"compiler":version,"flags":FLAGS,
                    "sha256":hashlib.sha256(target.read_bytes()).hexdigest()}


class PaddedKernel:
    def __init__(self, lib, store, operation):
        self.lib,self.store,self.op = lib,store,operation
        self.args = (int(store.kind=="sell"),OPERATIONS.index(operation),store.n,
                     store.size,store.groups,len(store.arrays[4]),
                     *(array.ctypes.data for array in store.arrays))

    def __call__(self, x, y=None, slope=.2):
        shape = (self.store.n,H,F) if self.op=="aggregation" else (self.store.e,H,1)
        output = torch.empty(shape,dtype=torch.float32)
        maxima = sums = None
        if self.op=="softmax":
            maxima = torch.empty((self.store.n,H),dtype=torch.float32)
            sums = torch.empty_like(maxima)
        self.lib.padded_run(*self.args,x.data_ptr(),y.data_ptr() if y is not None else None,
                            slope,output.data_ptr(),maxima.data_ptr() if maxima is not None else None,
                            sums.data_ptr() if sums is not None else None)
        return output
