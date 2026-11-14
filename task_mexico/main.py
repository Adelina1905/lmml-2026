"""Ruta de las Identidades - record linkage for LMML 2026 (task_mexico).

Usage:
    python main.py [input.csv] [output.json]

* input.csv   - CSV with columns source_table, record_id, first_name, last_name, address.
                Default: the first "*input*.csv" found in the current folder, else data/train-input.csv.
* output.json - where to write {"matches": [[id1, id2], ...]}. Default: the input file name
                with "input" replaced by "output" and .csv by .json, in the current folder
                (test-input.csv -> test-output.json); output.json if the name has no "input".

The JSON is also printed to stdout; progress messages go to stderr.
Only the Python standard library is used.

Method (details in EXPLANATION.md):
1. Blocking key = the multiset of digit groups in the address (street number, interior / building /
   flat numbers, postal code). The registration desks rewrite the address text (abbreviations,
   accents, word order) but never change these numbers, so duplicates share the key.
2. Inside a block, two records are linked if their names are compatible: name parts may be swapped,
   shortened to an initial, mistyped, have accents dropped or be a common nickname.
3. Safety net: a strict fuzzy pass links two still-separate groups only when the postal code is the
   same, all other numbers but one are equal and the differing one is a one-character typo, and
   both name parts agree. It is designed to fire only if the hidden data contains a typo in a number.
4. Links are merged into groups (union-find) and every pair inside a group is written out.
"""
import csv
import glob
import itertools
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher

# Common Spanish given-name nicknames (both directions are handled).
NICKNAMES = {
    "pepe": "jose", "pepito": "jose", "chepe": "jose", "jose": "jose",
    "paco": "francisco", "pancho": "francisco", "kiko": "francisco", "curro": "francisco",
    "nacho": "ignacio", "lalo": "eduardo", "memo": "guillermo", "beto": "alberto",
    "chucho": "jesus", "chuy": "jesus", "toño": "antonio", "tono": "antonio", "tonio": "antonio",
    "lupe": "guadalupe", "lupita": "guadalupe", "pili": "pilar", "conchita": "concepcion",
    "concha": "concepcion", "chela": "graciela", "lola": "dolores", "lolita": "dolores",
    "charo": "rosario", "maru": "maria", "mari": "maria", "nando": "fernando", "fer": "fernando",
    "rafa": "rafael", "quique": "enrique", "kike": "enrique", "manolo": "manuel", "manu": "manuel",
    "santi": "santiago", "chava": "salvador", "chente": "vicente", "goyo": "gregorio",
    "tere": "teresa", "paty": "patricia", "pati": "patricia", "vero": "veronica",
    "lucho": "luis", "pita": "guadalupe", "rodo": "rodolfo", "fito": "adolfo", "poncho": "alfonso",
    "licha": "alicia", "chabela": "isabel", "isa": "isabel", "nena": "elena", "cuca": "refugio",
    "cuquita": "refugio", "lencho": "lorenzo", "tavo": "gustavo", "panchito": "francisco",
}


def log(*a):
    print(*a, file=sys.stderr)


def strip_accents(s):
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def name_tokens(s):
    return re.sub(r"[^a-z ]", " ", strip_accents(s).lower()).split()


def address_numbers(addr):
    return tuple(sorted(re.findall(r"\d+", addr)))


def postal_code(addr):
    m = re.search(r"(\d{5})(?:-\d{4})?\s*$", addr.strip())
    return m.group(1) if m else None


def part_compatible(x, y):
    """Can name part x (token list) and name part y be the same name?"""
    if not x or not y:
        return False
    a, b = " ".join(x), " ".join(y)
    if a == b:
        return True
    # initial vs full name ("h" vs "hugo", "j m" vs "jose maria")
    if len(a.replace(" ", "")) <= 2 or len(b.replace(" ", "")) <= 2 or min(len(t) for t in x) == 1 or min(len(t) for t in y) == 1:
        ia = "".join(t[0] for t in x)
        ib = "".join(t[0] for t in y)
        if ia == ib or ia[0] == ib[0] and (len(x) == 1 or len(y) == 1):
            return True
    # nicknames
    na, nb = NICKNAMES.get(a, a), NICKNAMES.get(b, b)
    if na == nb:
        return True
    # typos / truncations / letter swaps
    if SequenceMatcher(None, a, b).ratio() >= 0.7:
        return True
    if a in b or b in a:
        return len(min(a, b, key=len)) >= 3
    return False


def name_score(r1, r2):
    """Number of name parts (0, 1 or 2) that agree, under the best of straight / swapped order."""
    f1, l1, f2, l2 = r1["fn"], r1["ln"], r2["fn"], r2["ln"]
    straight = part_compatible(f1, f2) + part_compatible(l1, l2)
    swapped = part_compatible(f1, l2) + part_compatible(l1, f2)
    return max(straight, swapped)


def one_char_typo(a, b):
    """True if digit strings a and b differ by one substitution, insertion, deletion or adjacent swap."""
    if a == b:
        return False
    if len(a) == len(b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        if len(diff) == 1:
            return True
        if len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]:
            return True
        return False
    if abs(len(a) - len(b)) == 1:
        s, l = (a, b) if len(a) < len(b) else (b, a)
        return any(l[:i] + l[i + 1:] == s for i in range(len(l)))
    return False


def numbers_one_typo(n1, n2):
    """Number multisets equal except one element that is a one-character typo of the other."""
    if len(n1) != len(n2) or n1 == n2:
        return False
    r1, r2 = list(n1), list(n2)
    for x in n1:
        if x in r2:
            r2.remove(x)
            r1.remove(x)
    return len(r1) == 1 and len(r2) == 1 and one_char_typo(r1[0], r2[0])


class DSU:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def load(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    recs = []
    for r in rows:
        r = {k.strip(): (v or "").strip() for k, v in r.items() if k is not None}
        recs.append({
            "id": r["record_id"],
            "table": r.get("source_table", ""),
            "fn": name_tokens(r.get("first_name", "")),
            "ln": name_tokens(r.get("last_name", "")),
            "nums": address_numbers(r.get("address", "")),
            "zip": postal_code(r.get("address", "")),
        })
    return recs


def link(recs, stats=None):
    stats = {} if stats is None else stats
    by_id = {r["id"]: r for r in recs}
    dsu = DSU(by_id)

    # 1) exact blocking on address numbers + lenient name check
    blocks = defaultdict(list)
    for r in recs:
        if r["nums"]:
            blocks[r["nums"]].append(r)
    linked = rejected = 0
    for members in blocks.values():
        for a, b in itertools.combinations(members, 2):
            if name_score(a, b) >= 1:
                dsu.union(a["id"], b["id"])
                linked += 1
            else:
                rejected += 1
                log("  name check rejected same-address pair:", a["id"], b["id"])
    stats["block_pairs_linked"] = linked
    stats["block_pairs_rejected"] = rejected

    # 2) strict fuzzy safety net for a typo in one address number
    def tables(root):
        return [by_id[i]["table"] for i in groups[root]]

    groups = defaultdict(list)
    for i in by_id:
        groups[dsu.find(i)].append(i)
    by_zip = defaultdict(list)
    for r in recs:
        if r["zip"]:
            by_zip[r["zip"]].append(r)
    fuzzy = 0
    for members in by_zip.values():
        for a, b in itertools.combinations(members, 2):
            ra, rb = dsu.find(a["id"]), dsu.find(b["id"])
            if ra == rb:
                continue
            if not numbers_one_typo(a["nums"], b["nums"]):
                continue
            if name_score(a, b) < 2:
                continue
            if set(tables(ra)) & set(tables(rb)):  # a person appears at most once per desk
                continue
            log("  fuzzy safety net linked:", a["id"], b["id"])
            dsu.union(ra, rb)
            groups[dsu.find(rb)] = groups.pop(ra, []) + groups.pop(rb, [])
            fuzzy += 1
    stats["fuzzy_links"] = fuzzy

    # 3) all pairs within each final group
    final = defaultdict(list)
    for i in by_id:
        final[dsu.find(i)].append(i)
    matches = []
    sizes = defaultdict(int)
    for members in final.values():
        sizes[len(members)] += 1
        members.sort()
        tb = [by_id[i]["table"] for i in members]
        if len(set(tb)) != len(tb):
            log("  note: group with repeated source table:", members)
        matches.extend([a, b] for a, b in itertools.combinations(members, 2))
    matches.sort()
    stats["group_sizes"] = dict(sorted(sizes.items()))
    stats["pairs"] = len(matches)
    return matches


def default_paths(argv):
    if len(argv) > 1:
        inp = argv[1]
    else:
        cands = sorted(glob.glob("*input*.csv"))
        cands.sort(key=lambda p: (not p.startswith("test"), p))
        inp = cands[0] if cands else os.path.join("data", "train-input.csv")
    if len(argv) > 2:
        out = argv[2]
    else:
        base = os.path.basename(inp)
        if "input" in base:
            out = os.path.splitext(base.replace("input", "output"))[0] + ".json"
        else:
            out = "output.json"
        if os.path.abspath(out) == os.path.abspath(inp):
            out = "output.json"
    return inp, out


def main():
    inp, out = default_paths(sys.argv)
    log(f"reading {inp}")
    recs = load(inp)
    stats = {}
    matches = link(recs, stats)
    result = {"matches": matches}
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh)
    log(f"{len(recs)} records -> {len(matches)} pairs; stats: {stats}")
    log(f"wrote {out}")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
