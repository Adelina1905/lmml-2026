"""task_cuba (Zemí): build, evaluate and package the retrieval submission.

    python solve.py            # full pipeline: model -> embeddings -> index -> evaluation -> submission.zip
    python solve.py --check    # only re-run local_check.py on submission.zip

Steps
 1. Get the neural model: bge-small-en-v1.5, int8 ONNX, from the npm package
    impro-plugin-semantic-mute-model@1.0.1 (HuggingFace is not reachable from every machine).
    Both files are checked against SHA-256 hashes.
 2. Cut every document into 200-word chunks (50-word overlap) and embed each chunk with the model
    (cached in cache/; this is the slow step, about 50 minutes on 2 CPU cores). Embed the titles too.
 3. Build index/: BM25 postings for 300-word chunks and for whole documents (uint8 weights,
    gap-encoded ids, compressed), int8 chunk and title embeddings, the model and its tokenizer.
 4. Evaluate on the 268 public questions: each signal alone, a grid over fusion weights,
    repeated 2-fold cross-validation of the weight choice, a typo stress test and ablations.
 5. Write submission/ (main.py + index/), zip it to submission.zip and run local_check.py on it.

Runtime of the submission needs only numpy, onnxruntime, tokenizers and PyStemmer, which are in the
grader's environment. Building also needs scipy and matplotlib.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from collections import Counter

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import main as M  # the submission's own text normalisation and stopword lists  # noqa: E402

DATA = os.path.join(HERE, "data")
CACHE = os.path.join(HERE, "cache")
MODEL_DIR = os.path.join(HERE, "models", "bge")
SUB = os.path.join(HERE, "submission")
RES = os.path.join(HERE, "results")

NPM_URL = "https://registry.npmjs.org/impro-plugin-semantic-mute-model/-/impro-plugin-semantic-mute-model-1.0.1.tgz"
MODEL_FILES = {  # name inside the npm package -> (local name, sha256)
    "package/bge-small-en-v1.5-q8.onnx": ("model.onnx", "bf490cda80c5d8b2c8068e60f37304638846cdb1fd124886e42046a7bd56cc63"),
    "package/bge-small-en-v1.5-tokenizer.json": ("tokenizer.json", "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"),
}
DENSE_CHUNK, DENSE_OVERLAP = 200, 50      # chunks for the neural model (fits its 512-token window)
LEX_CHUNK, LEX_OVERLAP = 300, 100         # chunks for BM25
K1, B = 1.5, 0.75                         # BM25 constants
CONFIG = {"typo_min_freq": 3, "query_prefix": "",
          "w_lex_chunk": 1.0, "w_lex_doc": 0.6, "w_dense_chunk": 1.5, "w_dense_title": 0.5}


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- data
def load_data():
    meta = [json.loads(l) for l in open(os.path.join(DATA, "corpus_meta.jsonl"), encoding="utf-8")]
    texts = {m["id"]: open(os.path.join(DATA, "corpus", m["id"] + ".txt"), encoding="utf-8").read() for m in meta}
    qs = [json.loads(l) for l in open(os.path.join(DATA, "sample-input.jsonl"), encoding="utf-8") if l.strip()]
    gold = {}
    for l in open(os.path.join(DATA, "sample-output.jsonl"), encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            gold[r["qid"]] = r["doc_id"]
    return meta, texts, qs, gold


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    w = text.split()
    if len(w) <= size:
        return [" ".join(w)]
    out, step = [], size - overlap
    for s in range(0, len(w), step):
        out.append(" ".join(w[s:s + size]))
        if s + size >= len(w):
            break
    return out


def clean_title(t: str) -> str:
    """'CHAPTER XVI. THE FEEDING OF THE HUNGRY' -> 'THE FEEDING OF THE HUNGRY'."""
    t = re.sub(r"^\s*CHAPTER\s*[:.]?\s*[IVXLCDMivxlcdm1l\d]*[A-Za-z]?\s*[.:]?\s*", "", t)
    t = re.sub(r"\(part \d of \d\)", "", t)
    return t.strip() or "chapter"


# ---------------------------------------------------------------- step 1: model
def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def get_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if all(os.path.exists(os.path.join(MODEL_DIR, n)) and sha256(os.path.join(MODEL_DIR, n)) == h
           for n, h in MODEL_FILES.values()):
        log("[1] model already present, hashes OK")
        return
    log("[1] downloading", NPM_URL)
    data = urllib.request.urlopen(NPM_URL, timeout=120).read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        for member, (name, h) in MODEL_FILES.items():
            blob = tf.extractfile(member).read()
            assert hashlib.sha256(blob).hexdigest() == h, f"hash mismatch for {member}"
            open(os.path.join(MODEL_DIR, name), "wb").write(blob)
    log("[1] model saved, hashes OK")


class Encoder:
    """bge-small-en-v1.5: [CLS] vector of the last layer, L2-normalised."""

    def __init__(self, threads: int = 2):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.s = ort.InferenceSession(os.path.join(MODEL_DIR, "model.onnx"), so, providers=["CPUExecutionProvider"])
        self.t = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
        self.t.enable_truncation(512)
        self.t.enable_padding()

    def encode(self, texts: list[str], bs: int = 16) -> np.ndarray:
        out = []
        for i in range(0, len(texts), bs):
            es = self.t.encode_batch(texts[i:i + bs])
            ids = np.array([e.ids for e in es], np.int64)
            am = np.array([e.attention_mask for e in es], np.int64)
            h = self.s.run(None, {"input_ids": ids, "attention_mask": am, "token_type_ids": np.zeros_like(ids)})[0]
            v = h[:, 0]
            out.append(v / np.linalg.norm(v, axis=1, keepdims=True))
        return np.concatenate(out).astype(np.float32)


# ---------------------------------------------------------------- step 2: embeddings
def embeddings(meta, texts):
    os.makedirs(CACHE, exist_ok=True)
    ids = [m["id"] for m in meta]
    chunks = [c for i in ids for c in chunk_words(texts[i], DENSE_CHUNK, DENSE_OVERLAP)]
    fv = os.path.join(CACHE, f"bge_chunks_{DENSE_CHUNK}_{DENSE_OVERLAP}.npy")
    ft = os.path.join(CACHE, "bge_titles.npy")
    enc = None
    if os.path.exists(fv) and np.load(fv, mmap_mode="r").shape[0] == len(chunks):
        log(f"[2] using cached chunk embeddings {fv}")
        V = np.load(fv)
        # spot-check that the cache was produced by this encoder
        enc = Encoder()
        idx = np.random.default_rng(0).choice(len(chunks), 32, replace=False)
        # int8 activations are quantised per batch, so vectors depend slightly on batch-mates:
        # compare by cosine similarity, not exact equality
        cos = (enc.encode([chunks[i] for i in idx]) * V[idx]).sum(1).min()
        log(f"[2] cache spot-check on 32 chunks: lowest cosine similarity = {cos:.4f}")
        assert cos > 0.99
    else:
        enc = Encoder()
        log(f"[2] embedding {len(chunks)} chunks (slow) ...")
        order = np.argsort([len(c) for c in chunks])   # similar lengths in a batch -> less padding
        V = np.zeros((len(chunks), 384), np.float32)
        t0 = time.time()
        for s in range(0, len(order), 256):
            sel = order[s:s + 256]
            V[sel] = enc.encode([chunks[i] for i in sel])
            log(f"    {s + len(sel)}/{len(chunks)}  {time.time() - t0:.0f}s")
        np.save(fv, V)
    if os.path.exists(ft):
        T = np.load(ft)
    else:
        T = (enc or Encoder()).encode([clean_title(m["title"]) for m in meta], bs=64)
        np.save(ft, T)
    return V, T


# ---------------------------------------------------------------- step 3: index
def build_index(meta, texts, V, T, out_dir):
    import Stemmer
    stem = Stemmer.Stemmer("english")
    os.makedirs(out_dir, exist_ok=True)
    ids = [m["id"] for m in meta]
    title = {m["id"]: m["title"] for m in meta}
    toks = lambda t: stem.stemWords([w for w in M.normalise(t).split() if w not in M.STOP])

    ctoks, cstart = [], []
    for i in ids:
        cstart.append(len(ctoks))
        ctoks += [toks(c) for c in chunk_words(texts[i], LEX_CHUNK, LEX_OVERLAP)]
    dtoks = [toks(texts[i] + " " + title[i]) for i in ids]
    terms: dict[str, int] = {}
    for d in dtoks:
        for t in d:
            terms.setdefault(t, len(terms))

    def bm25(docs):
        rows, cols, vals = [], [], []
        lens = np.array([len(d) for d in docs], np.float32)
        avg = lens.mean()
        for j, d in enumerate(docs):
            for t, c in Counter(d).items():
                rows.append(terms[t]); cols.append(j); vals.append(c)
        tf = sp.csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=(len(terms), len(docs)))
        df = np.diff(tf.indptr).astype(np.float32)
        n = len(docs)
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        tf.sort_indices()
        coo = tf.tocoo()
        part = coo.data * (K1 + 1) / (coo.data + K1 * (1 - B + B * lens[coo.col] / avg))  # in (0, K1+1)
        scale = np.float32(K1 + 1)
        q = np.clip(np.round(part / scale * 255), 1, 255).astype(np.uint8)
        return tf.indptr.astype(np.int32), coo.col, q, idf, scale

    def gaps(indptr, idx):
        d = np.diff(idx.astype(np.int64), prepend=0)
        first = indptr[:-1][np.diff(indptr) > 0]
        d[first] = idx[first]
        assert d.min() >= 0 and d.max() < 65536
        return d.astype(np.uint16)

    ci, cj, cv, cidf, cs = bm25(ctoks)
    di, dj, dv, didf, ds = bm25(dtoks)
    assert len(ctoks) < 65536
    words = Counter()
    for i in ids:
        words.update(M.normalise(texts[i]).split())
    words = {w: f for w, f in words.items() if not w.isdigit() and len(w) >= 3}
    wp = [w for w in json.load(open(os.path.join(MODEL_DIR, "tokenizer.json"), encoding="utf-8"))["model"]["vocab"] if w.isalpha()]
    enc = lambda L: np.frombuffer("\n".join(L).encode("utf-8"), np.uint8)
    np.savez_compressed(os.path.join(out_dir, "lex.npz"), doc_ids=enc(ids), terms=enc(list(terms)), words=enc(list(words)),
                        word_freq=np.array(list(words.values()), np.int32), wordpieces=enc(wp),
                        c_indptr=ci, c_idx=gaps(ci, cj), c_val=cv, c_idf=cidf, c_scale=np.array([cs], np.float32),
                        c_start=np.array(cstart, np.int32), n_chunks=np.array([len(ctoks)]),
                        d_indptr=di, d_idx=gaps(di, dj), d_val=dv, d_idf=didf, d_scale=np.array([ds], np.float32))
    vstart, n = [], 0
    for i in ids:
        vstart.append(n)
        n += len(chunk_words(texts[i], DENSE_CHUNK, DENSE_OVERLAP))
    assert n == len(V)

    def q8(X):
        s = np.abs(X).max(1) / 127.0
        return np.round(X / s[:, None]).astype(np.int8), s.astype(np.float32)

    v, vs = q8(V)
    t, ts = q8(T)
    np.savez(os.path.join(out_dir, "dense.npz"), v=v, v_scale=vs, v_start=np.array(vstart, np.int32), t=t, t_scale=ts)
    shutil.copy(os.path.join(MODEL_DIR, "model.onnx"), os.path.join(out_dir, "model.onnx"))
    with open(os.path.join(MODEL_DIR, "tokenizer.json"), "rb") as f, \
            gzip.GzipFile(os.path.join(out_dir, "tokenizer.json.gz"), "wb", 9, mtime=0) as g:
        g.write(f.read())
    json.dump(CONFIG, open(os.path.join(out_dir, "config.json"), "w"), indent=1)
    sizes = {f: os.path.getsize(os.path.join(out_dir, f)) / 1e6 for f in sorted(os.listdir(out_dir))}
    return sizes, len(ctoks), len(V)


# ---------------------------------------------------------------- step 4: evaluation
def evaluate(index_dir, qs, gold, report):
    r = M.Retriever(index_dir)
    G = np.array([r.doc_ids.index(gold[q["qid"]]) for q in qs])
    names = ["lex_chunk", "lex_doc", "dense_chunk", "dense_title"]
    comps = []
    for q in qs:
        terms, dt = r.process(q["question"])
        cs = r.bm25(terms, r.c_indptr, r.c_idx, r.c_val, r.c_idf, r.n_chunks)
        lc = np.maximum.reduceat(cs, r.c_start)
        ld = r.bm25(terms, r.d_indptr, r.d_idx, r.d_val, r.d_idf, r.n_docs)
        qv = r.encode(dt)
        dc = np.maximum.reduceat(r.v @ qv, r.v_start)
        tt = r.t @ qv
        comps.append([r.norm(x) for x in (lc, ld, dc, tt)])
    C = np.array(comps)

    def hits(w, k=3):
        S = np.einsum("qcd,c->qd", C, np.array(w, np.float32))
        top = np.argsort(-S, 1)[:, :k]
        return (top == G[:, None]).any(1)

    report.append("\n## Each signal alone (public questions, n=%d)" % len(qs))
    report.append(f"{'signal':<14}{'hit@1':>8}{'hit@3':>8}{'hit@10':>8}")
    single = {}
    for i, n in enumerate(names):
        w = [0.0] * 4
        w[i] = 1.0
        single[n] = [hits(w, k).mean() for k in (1, 3, 10)]
        report.append(f"{n:<14}" + "".join(f"{x:>8.3f}" for x in single[n]))

    grid = list(itertools.product([1.0], [0, 0.3, 0.6, 1.0], [0.5, 0.75, 1, 1.5, 2, 3], [0, 0.15, 0.3, 0.5, 0.75]))
    H = np.array([hits(w) for w in grid])
    with open(os.path.join(RES, "fusion_grid.csv"), "w") as f:
        f.write("w_lex_chunk,w_lex_doc,w_dense_chunk,w_dense_title,hit_at_3\n")
        for w, h in sorted(zip(grid, H.mean(1)), key=lambda x: -x[1]):
            f.write(",".join(str(x) for x in w) + f",{h:.4f}\n")
    chosen = (CONFIG["w_lex_chunk"], CONFIG["w_lex_doc"], CONFIG["w_dense_chunk"], CONFIG["w_dense_title"])
    hc = H[grid.index(chosen)]
    report.append("\n## Fusion weights")
    report.append(f"grid of {len(grid)} weight settings; best hit@3 = {H.mean(1).max():.3f}; "
                  f"settings within 0.01 of best: {(H.mean(1) >= H.mean(1).max() - 0.01).sum()}")
    report.append(f"chosen {dict(zip(names, chosen))}: hit@3 = {hc.mean():.3f}")
    rng = np.random.default_rng(0)
    outs = []
    for _ in range(200):
        p = rng.permutation(len(G))
        a, b = p[:len(G) // 2], p[len(G) // 2:]
        outs.append(H[np.argmax(H[:, a].mean(1)), b].mean())
    boot = [hc[rng.integers(0, len(hc), len(hc))].mean() for _ in range(2000)]
    report.append(f"repeated 2-fold CV (pick weights on one half, score the other half, 200 splits): "
                  f"mean {np.mean(outs):.3f}, 5th percentile {np.percentile(outs, 5):.3f}, worst {np.min(outs):.3f}")
    report.append(f"bootstrap of the chosen setting: 5th percentile {np.percentile(boot, 5):.3f}")

    # typo stress test and ablations, through the real search() path
    import random
    rnd = random.Random(7)

    def typo(w):
        i = rnd.randrange(1, len(w) - 1)
        op = rnd.choice(["swap", "del", "dup", "rep"])
        if op == "swap":
            return w[:i] + w[i + 1] + w[i] + w[i + 2:]
        if op == "del":
            return w[:i] + w[i + 1:]
        if op == "dup":
            return w[:i] + w[i] + w[i:]
        return w[:i] + rnd.choice("aeiourstn") + w[i + 1:]

    def perturb(q):
        ws = q.split()
        cand = [i for i, w in enumerate(ws) if w.isalpha() and len(w) >= 5]
        for i in rnd.sample(cand, min(3, len(cand))):
            ws[i] = typo(ws[i])
        return " ".join(ws)

    noisy = [perturb(q["question"]) for q in qs]
    run = lambda Q: np.mean([gold[q["qid"]] in r.search(x, 3) for q, x in zip(qs, Q)])
    clean = [q["question"] for q in qs]
    report.append("\n## Robustness and ablations (hit@3)")
    report.append(f"full system, original questions          {run(clean):.3f}")
    report.append(f"full system, 3 random typos per question {run(noisy):.3f}")
    corr = r.correct
    r.correct = lambda w: w
    report.append(f"no typo correction, original questions   {run(clean):.3f}")
    report.append(f"no typo correction, 3 typos per question {run(noisy):.3f}")
    r.correct = corr
    cfg = dict(r.cfg)
    r.cfg = dict(cfg, w_dense_chunk=0.0, w_dense_title=0.0)
    report.append(f"lexical only (no neural model)           {run(clean):.3f}")
    r.cfg = dict(cfg, w_lex_chunk=0.0, w_lex_doc=0.0)
    report.append(f"neural only (no BM25)                    {run(clean):.3f}")
    r.cfg = cfg

    # misses of the final system
    misses = [(q, r.search(q["question"], 3)) for q in qs if gold[q["qid"]] not in r.search(q["question"], 3)]
    with open(os.path.join(RES, "misses.txt"), "w", encoding="utf-8") as f:
        for q, top in misses:
            f.write(f"{q['qid']}  gold={gold[q['qid']]}  top3={top}\n    {q['question']}\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = names + ["fused (final)"]
        vals = [single[n][1] for n in names] + [hc.mean()]
        fig, ax = plt.subplots(figsize=(7, 3.6))
        bars = ax.bar(labels, vals, color=["#8aa1c1"] * 4 + ["#2f6b3f"])
        ax.axhline(0.8, color="#b23b3b", ls="--", lw=1)
        ax.text(-0.45, 0.815, "limit 0.80", color="#b23b3b", ha="left", va="bottom", fontsize=8)
        for b_, v in zip(bars, vals):
            ax.text(b_.get_x() + b_.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("hit@3 on public questions")
        ax.set_title("Each signal alone vs. the fused ranking")
        fig.tight_layout()
        fig.savefig(os.path.join(RES, "hit_at_3_by_signal.png"), dpi=130)
    except Exception as e:  # plotting is optional
        report.append(f"(plot skipped: {e})")


# ---------------------------------------------------------------- step 5: package and check
def package():
    zpath = os.path.join(HERE, "submission.zip")
    if os.path.exists(zpath):
        os.remove(zpath)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(os.path.join(SUB, "main.py"), "main.py")
        for f in sorted(os.listdir(os.path.join(SUB, "index"))):
            z.write(os.path.join(SUB, "index", f), "index/" + f)
    return zpath


def local_check(zpath):
    out = subprocess.run([sys.executable, os.path.join(HERE, "local_check.py"), zpath],
                         capture_output=True, text=True, encoding="utf-8")
    text = out.stdout + out.stderr
    open(os.path.join(RES, "local_check.txt"), "w", encoding="utf-8").write(text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only run local_check.py on submission.zip")
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    if args.check:
        log(local_check(os.path.join(HERE, "submission.zip")))
        return
    report = ["# task_cuba (Zemí) build report"]
    meta, texts, qs, gold = load_data()
    get_model()
    V, T = embeddings(meta, texts)
    if os.path.exists(SUB):
        shutil.rmtree(SUB)
    os.makedirs(os.path.join(SUB, "index"))
    shutil.copy(os.path.join(HERE, "main.py"), os.path.join(SUB, "main.py"))
    sizes, n_lex, n_dense = build_index(meta, texts, V, T, os.path.join(SUB, "index"))
    report.append(f"\n## Index ({len(meta)} documents, {n_lex} BM25 chunks of {LEX_CHUNK} words, "
                  f"{n_dense} neural chunks of {DENSE_CHUNK} words)")
    for f, s in sizes.items():
        report.append(f"{f:<20}{s:>8.2f} MB")
    total = sum(sizes.values()) + os.path.getsize(os.path.join(SUB, "main.py")) / 1e6
    report.append(f"{'main.py + index/':<20}{total:>8.2f} MB")
    log("\n".join(report))
    log("[4] evaluating ...")
    evaluate(os.path.join(SUB, "index"), qs, gold, report)
    zpath = package()
    log("[5] running local_check.py on", zpath)
    lc = local_check(zpath)
    report.append("\n## local_check.py on submission.zip\n" + lc.strip())
    open(os.path.join(RES, "report.md"), "w", encoding="utf-8").write("\n".join(report) + "\n")
    log("\n".join(report))


if __name__ == "__main__":
    main()
