"""Compile a small OpenMP C ABI library, outside the checkout's source tree."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter
import torch

HERE=Path(__file__).resolve().parent
FLAGS=['-std=c++17','-O3','-fPIC','-shared','-fopenmp','-ffp-contract=off']


def build():
    compiler=os.environ.get('CXX','g++')
    version=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0]
    identity=hashlib.sha256((HERE/'kernels.cpp').read_bytes()+json.dumps([version,FLAGS]).encode()).hexdigest()
    directory=Path(os.environ.get('TUNING_AXES_BUILD_DIR','results/tuning_axes_build')).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    target=directory/f'libaxes-{identity[:16]}.so'
    cached=target.exists();start=perf_counter()
    if not cached:
        temp=target.with_suffix('.tmp.so')
        subprocess.run([compiler,*FLAGS,str(HERE/'kernels.cpp'),'-o',str(temp)],check=True,capture_output=True,text=True)
        temp.replace(target)
    info={'compiler':version,'flags':FLAGS,'source_sha256':hashlib.sha256((HERE/'kernels.cpp').read_bytes()).hexdigest(),
                   'binary_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'build_seconds':perf_counter()-start,'cache_hit':cached}
    manifest=target.with_suffix('.build.json')
    if not cached:manifest.write_text(json.dumps(info,indent=2)+'\n')
    info['initial_compilation']=json.loads(manifest.read_text()) if manifest.exists() else None
    return target,info


class Native:
    def __init__(self):
        path,self.build_info=build();self.lib=C.CDLL(str(path))
        P=C.c_void_p;I=C.c_int;L=C.c_int64
        signatures={'axes_team_size':[I], 'axes_node':[P]*5+[L,I,I,I,I],
            'axes_add':[P]*5+[L,I,I,I], 'axes_leaky':[P,P,L,C.c_float,I],
            'axes_softmax':[P]*4+[L,L,I,I,P,P,I,P],
            'axes_aggregate':[P]*6+[L,L,I,I,I,I,P,P,I,P]}
        for name,args in signatures.items():
            function=getattr(self.lib,name);function.argtypes=args;function.restype=I if name=='axes_team_size' else None

    @staticmethod
    def ptr(tensor):return None if tensor is None else C.c_void_p(tensor.data_ptr())

    def node(self,z,al,ar,n,h,f,layout,threads):
        l=torch.empty((n,h,1));r=torch.empty_like(l)
        self.lib.axes_node(*map(self.ptr,[z,al,ar,l,r]),n,h,f,layout,threads)
        return l,r

    def add(self,l,r,src,dst,e,h,layout,threads):
        q=torch.empty((h,e) if layout else (e,h))
        self.lib.axes_add(*map(self.ptr,[l,r,src,dst,q]),e,h,layout,threads)
        return q

    def leaky(self,q,slope,threads):
        s=torch.empty_like(q);self.lib.axes_leaky(self.ptr(q),self.ptr(s),q.numel(),slope,threads);return s

    def softmax(self,ptr,eids,s,n,e,h,layout,partition,threads,diagnostic=False):
        a=torch.empty_like(s);order,parts=partition['order'],partition['parts']
        times=torch.empty(threads,dtype=torch.float64) if diagnostic else None
        self.lib.axes_softmax(*map(self.ptr,[ptr,eids,s,a]),n,e,h,layout,self.ptr(order),self.ptr(parts),threads,self.ptr(times))
        return (a,times) if diagnostic else a

    def aggregate(self,ptr,src,eids,z,a,n,e,h,f,zlayout,elayout,partition,threads,diagnostic=False):
        y=torch.empty((n,h,f));order,parts=partition['order'],partition['parts']
        times=torch.empty(threads,dtype=torch.float64) if diagnostic else None
        self.lib.axes_aggregate(*map(self.ptr,[ptr,src,eids,z,a,y]),n,e,h,f,zlayout,elayout,self.ptr(order),self.ptr(parts),threads,self.ptr(times))
        return (y,times) if diagnostic else y


if __name__=='__main__':
    path,info=build();print(json.dumps({'library':str(path),**info},indent=2))
