"""
LMML 2026 - The Drunken Mario Adventures
Recovers the hidden word from Marathon Lab without having to watch the GUI.

Pipeline
  1. The .exe is a Nuitka "onefile" build. Its payload (PE resource RT_RCDATA id 27)
     is "KA" + "Y" + a zstd stream holding every bundled file
     (UTF-16 name, NUL, uint64 size, bytes).
  2. One of those files is main.dll: the compiled game. Its resource id 3 is
     Nuitka's constants blob, with one section per Python module
     (".lab.fitness", ".lab.level", ...).
  3. From lab.fitness we read the reward constants and from lab.level the 20
     coin columns (COIN_SITES). The disassembly of lab.fitness (see EXPLANATION)
     gives:
        _BASIS = 919247
        def _spread(sites):
            h = 0
            for i, x in enumerate(sites):
                h = (h * 131 + x * 7 + i) & 1048575      # 20-bit hash
            return h
        def _terms(sites):
            n = len(sites)
            return f"{_BASIS ^ _spread(sites):0{n}b}"  # one bit per coin
     A '1' gives that coin a positive reward (+COIN_WEIGHT): the trained agent
     goes out of its way to grab it. A '0' gives it a penalty: the agent walks
     past it.
  4. Read the take/skip pattern in groups of 5 bits: A=1 ... Z=26.

Usage:  python solve.py [path/to/marathon-lab-windows-x86_64.exe]
Needs:  pip install pefile zstandard
"""
import io
import os
import struct
import sys

import pefile
import zstandard

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "marathon-lab-windows-x86_64.exe")
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)
report = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    report.append(s)


# ---------------------------------------------------------------- 1. onefile payload
def resource(pe, type_id, name_id):
    for t in pe.DIRECTORY_ENTRY_RESOURCE.entries:
        if t.id != type_id:
            continue
        for n in t.directory.entries:
            if n.id == name_id:
                d = n.directory.entries[0].data.struct
                return pe.get_data(d.OffsetToData, d.Size)
    raise KeyError((type_id, name_id))


def onefile_files(exe_path):
    blob = resource(pefile.PE(exe_path), 10, 27)
    assert blob[:3] == b"KAY", "not a compressed Nuitka onefile payload"
    raw = zstandard.ZstdDecompressor().decompressobj().decompress(blob[3:])
    files, i = {}, 0
    while i < len(raw):
        j = i
        while raw[j:j + 2] != b"\0\0" or (j - i) % 2:
            j += 1
        name = raw[i:j].decode("utf-16le")
        i = j + 2
        if not name:
            break
        size = struct.unpack("<Q", raw[i:i + 8])[0]
        i += 8
        files[name] = raw[i:i + size]
        i += size
    return files


# ---------------------------------------------------------------- 2. Nuitka constants blob
class Blob:
    def __init__(self, data):
        self.d = data

    def varint(self, i):
        r, f = 0, 1
        while True:
            v = self.d[i]
            i += 1
            r += (v & 127) * f
            if v < 128:
                return r, i
            f <<= 7

    def cstr(self, i):
        j = self.d.index(b"\0", i)
        return self.d[i:j].decode("utf8", "replace"), j + 1

    def one(self, i):
        d, c = self.d, chr(self.d[i])
        i += 1
        if c in "TLPS":
            n, i = self.varint(i)
            items = []
            for _ in range(n):
                o, i = self.one(i)
                items.append(o)
            return (tuple(items) if c == "T" else items), i
        if c == "D":
            n, i = self.varint(i)
            ks, vs = [], []
            for _ in range(n):
                k, i = self.one(i)
                ks.append(k)
            for _ in range(n):
                v, i = self.one(i)
                vs.append(v)
            return dict(zip(map(repr, ks), vs)), i
        if c == "l":
            return self.varint(i)
        if c == "q":
            v, i = self.varint(i)
            return -v, i
        if c in "gG":
            n, i = self.varint(i)
            v = 0
            for k in range(n):
                dg, i = self.varint(i)
                v += dg << (30 * k)
            return (v if c == "g" else -v), i
        if c == "f":
            return struct.unpack("<d", d[i:i + 8])[0], i + 8
        if c == "Z":
            return [0.0, -0.0, float("inf"), float("-inf"), float("nan")][d[i]] if d[i] < 5 else ("Zfloat", d[i]), i + 1
        if c in "au":
            return self.cstr(i)
        if c == "v":
            n, i = self.varint(i)
            return d[i:i + n].decode("utf8", "replace"), i + n
        if c == "w":
            return chr(d[i]), i + 1
        if c == "s":
            return "", i
        if c == "b":
            n, i = self.varint(i)
            return d[i:i + n], i + n
        if c == "n":
            return None, i
        if c == "t":
            return True, i
        if c == "F":
            return False, i
        if c in ":;":
            a, i = self.one(i)
            b, i = self.one(i)
            e, i = self.one(i)
            return (("slice" if c == ":" else "range"), a, b, e), i
        if c == "p":
            return "<prev>", i
        if c == "E":
            return Ellipsis, i
        raise ValueError(f"unknown constant tag {c!r} at {i - 1}")

    def module(self, name):
        key = b"." + name.encode() + b"\0"
        s = self.d.index(key) + len(key)
        count = struct.unpack("<H", self.d[s + 4:s + 6])[0]
        i, out = s + 6, []
        for _ in range(count):
            o, i = self.one(i)
            out.append(o)
        return out


# ---------------------------------------------------------------- 3. rebuild the coin signs
def spread(sites, mult, mul_x, mask):
    h = 0
    for i, x in enumerate(sites):
        h = (h * mult + x * mul_x + i) & mask
    return h


def main():
    log("== Marathon Lab: hidden-word recovery ==")
    log("exe:", os.path.basename(EXE), os.path.getsize(EXE), "bytes")
    files = onefile_files(EXE)
    log("onefile payload:", len(files), "files; lab data:", [k for k in files if k.startswith("lab")])
    main_dll = pefile.PE(data=files["main.dll"])
    blob = Blob(resource(main_dll, 10, 3))

    fit = blob.module("lab.fitness")
    lvl = blob.module("lab.level")
    with open(os.path.join(RESULTS, "lab_fitness_constants.txt"), "w") as f:
        f.write("\n".join(f"{k:3d} {v!r}" for k, v in enumerate(fit)))
    with open(os.path.join(RESULTS, "lab_level_constants.txt"), "w") as f:
        f.write("\n".join(f"{k:3d} {v!r}" for k, v in enumerate(lvl)))

    # constants as they appear in lab.fitness (indices verified against the disassembly)
    mult, mul_x, mask = fit[1], fit[2], fit[3]          # 131, 7, 1048575
    basis = fit[34]                                     # _BASIS = 919247
    progress, time_cost, coin_w, death, win = fit[29], fit[30], fit[31], fit[32], fit[33]
    sites = next(v for v in lvl if isinstance(v, list) and len(v) == 20 and all(isinstance(x, int) for x in v))
    log(f"reward terms: PROGRESS={progress} TIME_COST={time_cost} COIN_WEIGHT={coin_w} DEATH={death} WIN={win}")
    log(f"hash: h = (h*{mult} + x*{mul_x} + i) & {mask}   _BASIS = {basis}")
    log("COIN_SITES (tile columns):", sites)

    h = spread(sites, mult, mul_x, mask)
    bits = f"{basis ^ h:0{len(sites)}b}"
    log(f"_spread(COIN_SITES) = {h}")
    log(f"_terms = _BASIS ^ _spread = {basis ^ h}  ->  bits {bits}")
    log("")
    log(" site  column  bit  sign  trained agent")
    for k, (x, b) in enumerate(zip(sites, bits)):
        log(f"  {k:2d}    {x:4d}    {b}    {'+' if b == '1' else '-'}    {'TAKES the coin' if b == '1' else 'walks past'}")

    letters = []
    for k in range(0, len(bits), 5):
        v = int(bits[k:k + 5], 2)
        letters.append(chr(64 + v) if 1 <= v <= 26 else "?")
        log(f"  bits {bits[k:k + 5]} = {v:2d} -> {letters[-1]}")
    word = "".join(letters)
    flag = f"SIGMOID_{word}"
    log("")
    log("WORD =", word)
    log("FLAG =", flag)

    with open(os.path.join(HERE, "FLAG.txt"), "w") as f:
        f.write(flag)
    with open(os.path.join(RESULTS, "report.txt"), "w") as f:
        f.write("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
