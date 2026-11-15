"""Recover the Island Duel star-chart flag from the supplied participant data.

The star chart that a qualifying run would save (.chart.json) ships inside the
engine binary. This script extracts it, then decodes it in three steps that
follow the chart's own verse:
  1. "The five lights of home keep the Greek watch." -> read the five large stars
     (the Southern Cross on Samoa's flag) in Bayer order alpha, beta, gamma,
     delta, epsilon Crucis.
  2. "Each keeps its nearer companions before the farther." -> every small star
     belongs to its nearest large star; after each large star, read its
     companions from nearest to farthest.
  3. "Our foundation turns the alphabet, over and over... undo those turns." ->
     Vigenere decryption with Samoa's motto "Fa'avae i le Atua Samoa"
     ("Samoa is founded on God") as the repeating key.
Run:  python solve_chart.py
"""
import json, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(HERE, "engine")


def extract_chart():
    for name in ("samoa-linux-x86_64", "samoa-windows-x86_64.exe"):
        path = os.path.join(ENGINE, name)
        if os.path.exists(path):
            data = open(path, "rb").read()
            i = data.find(b'{"prefix":"SIGMOID_"')
            if i >= 0:
                return json.loads(data[i:data.index(b"\x00", i)].decode("utf-8")), name
    sys.exit("chart data not found in engine binaries")


def bayer_order(guides):
    """Map the five large stars to alpha..epsilon Crucis by their position on the
    flag (sky orientation: gamma top, alpha bottom, beta left, delta right,
    epsilon = the smallest star)."""
    eps = min(guides, key=lambda s: s["size"])
    big = [g for g in guides if g is not eps]
    gamma = min(big, key=lambda s: s["y"])   # top (y grows downwards)
    alpha = max(big, key=lambda s: s["y"])   # bottom
    rest = [g for g in big if g not in (gamma, alpha)]
    beta = min(rest, key=lambda s: s["x"])   # left
    delta = max(rest, key=lambda s: s["x"])  # right
    return [alpha, beta, gamma, delta, eps]


def main():
    chart, src = extract_chart()
    lines = [f"chart extracted from engine/{src}", "verse:"] + ["  " + v for v in chart["verse"]]
    stars = chart["stars"]
    guides = [s for s in stars if s["size"] > 6]
    comps = [s for s in stars if s["size"] <= 6]
    dist = lambda a, b: math.hypot(a["x"] - b["x"], a["y"] - b["y"])
    order = bayer_order(guides)
    names = ["alpha", "beta", "gamma", "delta", "epsilon"]
    cipher = ""
    for name, g in zip(names, order):
        mine = sorted((dist(c, g), c["text"]) for c in comps
                      if min(guides, key=lambda h: dist(c, h)) is g)
        chunk = g["text"] + "".join(t for _, t in mine)
        cipher += chunk
        lines.append(f"{name:8s} Crucis = {g['text']} at ({g['x']},{g['y']}); companions by distance: "
                     + ", ".join(f"{t}@{d:.0f}" for d, t in mine) + f" -> {chunk}")
    key = "FAAVAEILEATUASAMOA"  # Fa'avae i le Atua Samoa, letters only
    plain = "".join(chr((ord(ch) - 65 - (ord(key[i % len(key)]) - 65)) % 26 + 65)
                    for i, ch in enumerate(cipher))
    flag = chart["prefix"] + plain
    lines += [f"ciphertext : {cipher}", f"key        : {key} (repeated)", f"plaintext  : {plain}", f"FLAG       : {flag}"]
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    open(os.path.join(HERE, "results", "chart_decode.txt"), "w").write("\n".join(lines) + "\n")
    open(os.path.join(HERE, "FLAG.txt"), "w").write(flag + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
