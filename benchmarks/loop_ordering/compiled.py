"""Build and inspect the controlled serial family, bind each C ABI specialization."""
import ctypes as C
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from design import SYMBOLS

HERE = Path(__file__).resolve().parent
FLAGS = ("-std=c++17","-O3","-fPIC","-shared","-ffp-contract=off",
         "-fno-loop-interchange","-fno-loop-unroll-and-jam")
P, I, Float = C.c_void_p, C.c_int64, C.c_float
SIGNATURES = {"A": [P]*5+[I], "B": [P]*5+[I,Float],
              "C": [P]*5+[I], "D": [P]*6+[I]}


def sha(path, *, text=False):
    raw = Path(path).read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n",b"\n") if text else raw).hexdigest()


def assembly(binary):
    names = subprocess.check_output(["nm","-DS","--defined-only",str(binary)],text=True)
    records = {}
    for code, symbol in SYMBOLS.items():
        address, size, kind, _ = next(line.split() for line in names.splitlines()
                                      if line.split()[-1]==symbol)
        assert kind=="T"
        body = subprocess.check_output(
            ["objdump","-d","--no-show-raw-insn",f"--disassemble={symbol}",str(binary)],text=True)
        instructions = re.findall(r"^\s*[0-9a-f]+:\s*([^\n]+)$",body,re.MULTILINE)
        # Remove absolute jump targets but retain symbol-relative destinations,
        # operands, local branch structure and call names.
        normal = [re.sub(r"\b[0-9a-f]+\s+<", "<", s.split("#")[0]).replace(symbol,"SELF").strip()
                  for s in instructions]
        assert normal
        records[code] = dict(symbol=symbol, address=int(address,16), size_bytes=int(size,16),
                             instructions=len(normal), code_sha256=hashlib.sha256(
                                 "\n".join(normal).encode()).hexdigest())
    return records


def compile_library():
    cc = os.environ.get("CXX","g++")
    version = subprocess.check_output([cc,"--version"],text=True).splitlines()[0]
    source = HERE/"kernels.cpp"
    key = hashlib.sha256((sha(source,text=True)+json.dumps([version,FLAGS])).encode()).hexdigest()
    path = Path("results/loop_order_build").resolve()/f"loops-{key[:16]}.so"
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        tmp = path.with_suffix(f".{os.getpid()}.tmp.so")
        subprocess.run([cc,*FLAGS,str(source),"-o",str(tmp)],check=True)
        tmp.replace(path)
    # Verify serial execution does not depend on an OpenMP runtime.
    dynamic = subprocess.check_output(["readelf","-d",str(path)],text=True)
    libraries = re.findall(r"\(NEEDED\).*\[([^\]]+)\]",dynamic)
    assert not any("libgomp" in name or "libomp" in name for name in libraries)
    return path,dict(compiler=version,flags=list(FLAGS),source_sha256=sha(source,text=True),
                     binary_sha256=sha(path),symbols=assembly(path),needed_libraries=libraries,
                     binutils=subprocess.check_output(["objdump","--version"],text=True).splitlines()[0])


class Compiled:
    def __init__(self, binary):
        self.library = C.CDLL(str(binary))
        self.functions = {}
        for code,symbol in SYMBOLS.items():
            function = getattr(self.library,symbol)
            function.argtypes = SIGNATURES[code[0]]
            function.restype = None
            self.functions[code] = function

    def offsets(self):
        addresses = {code: C.cast(fn,P).value for code,fn in self.functions.items()}
        return {code: address-addresses["A1"] for code,address in addresses.items()}

    def bind(self, method, code):
        # The first argument of this bound call is the actual C address chosen
        # at setup, not a mode flag or a timed dispatch branch.
        return partial(method, self.functions[code])
