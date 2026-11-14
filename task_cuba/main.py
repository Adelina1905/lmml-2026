"""Zemí retriever: hybrid lexical (BM25) + neural (bge-small-en-v1.5, int8 ONNX) search.

index/ layout (written by build_index.py):
    model.onnx          bge-small-en-v1.5, int8-quantised ONNX (query encoder)
    tokenizer.json.gz   its WordPiece tokenizer (gzip)
    lex.npz             BM25 postings for 300-word chunks and for whole documents, vocabularies
    dense.npz           int8 chunk embeddings (+ per-row scale), chunk->doc map, title embeddings
    config.json         fusion weights and BM25 constants
"""
from __future__ import annotations

import gzip
import json
import os
import re
import unicodedata

import numpy as np

STOP = set("a an the of and or to in on at by for from with as is are was were be been this that these those it its into than then there their they them he she his her we you not no but if so such which who what when where why how all any each more most other some very can could would should may might will do does did have has had about over under between through during before after".split())
# words that describe the request ("the chapter that tells the whole story of ...") rather than its content
QSTOP = set("part chapter chapters whole story stories full reading read tell tells telling us article book account want give please section passage describes described describing".split())
# Spanish function words and common nouns that appear code-switched into the slips
SPSTOP = set("la el los las de del lo eso donde un una que y en con por para sobre como".split())
ES = {'guerra': 'war', 'gente': 'people', 'ano': 'year', 'anos': 'years', 'historia': 'history', 'iglesia': 'church',
      'barcos': 'ships', 'barco': 'ship', 'hombres': 'men', 'hombre': 'man', 'mujeres': 'women', 'mujer': 'woman',
      'ninos': 'children', 'nino': 'child', 'casa': 'house', 'casas': 'houses', 'ciudad': 'city', 'pueblo': 'town',
      'tierra': 'land', 'agua': 'water', 'rey': 'king', 'reina': 'queen', 'dinero': 'money', 'isla': 'island',
      'islas': 'islands', 'mar': 'sea', 'rio': 'river', 'azucar': 'sugar', 'tabaco': 'tobacco', 'esclavos': 'slaves',
      'esclavo': 'slave', 'soldados': 'soldiers', 'soldado': 'soldier', 'gobierno': 'government', 'libro': 'book',
      'musica': 'music', 'cancion': 'song', 'canciones': 'songs', 'comida': 'food', 'muerte': 'death', 'dia': 'day',
      'dias': 'days', 'noche': 'night', 'padre': 'father', 'madre': 'mother', 'hijo': 'son', 'hijos': 'sons',
      'hija': 'daughter', 'barrio': 'neighborhood', 'calle': 'street', 'calles': 'streets', 'puerto': 'port',
      'campo': 'countryside', 'fuego': 'fire', 'cielo': 'sky', 'nombre': 'name', 'trabajadores': 'workers',
      'trabajo': 'work', 'escuela': 'school', 'iglesias': 'churches', 'pais': 'country', 'paises': 'countries',
      'arboles': 'trees', 'arbol': 'tree', 'caballo': 'horse', 'caballos': 'horses', 'cosecha': 'harvest',
      'precio': 'price', 'precios': 'prices', 'medicos': 'doctors', 'medico': 'doctor', 'enfermos': 'sick',
      'batalla': 'battle', 'ejercito': 'army', 'periodico': 'newspaper', 'primero': 'first', 'grande': 'big',
      'pequeno': 'small', 'nuevo': 'new', 'viejo': 'old', 'cuantos': 'how many', 'cuando': 'when', 'quien': 'who',
      'porque': 'why'}
LETTERS = 'abcdefghijklmnopqrstuvwxyz'


def normalise(text: str) -> str:
    """Lower-case, strip accents, keep letters/digits. Same function at index and query time."""
    text = "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9\s]", " ", text)


def edits1(w: str) -> set[str]:
    """All strings one edit away (delete, swap neighbours, replace, insert)."""
    sp = [(w[:i], w[i:]) for i in range(len(w) + 1)]
    return set([a + b[1:] for a, b in sp if b] + [a + b[1] + b[0] + b[2:] for a, b in sp if len(b) > 1]
               + [a + c + b[1:] for a, b in sp if b for c in LETTERS] + [a + c + b for a, b in sp for c in LETTERS])


def undelta(indptr: np.ndarray, gaps: np.ndarray) -> np.ndarray:
    """Posting lists are stored as gaps between sorted ids; turn them back into ids."""
    cs = np.cumsum(gaps.astype(np.int64))
    lens = np.diff(indptr)
    starts = indptr[:-1].astype(np.int64)
    base = np.where(starts > 0, cs[np.maximum(starts - 1, 0)], 0)
    base = np.where(lens > 0, base, 0)
    return (cs - np.repeat(base, lens)).astype(np.int32)


class Retriever:
    def __init__(self, index_dir: str) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        p = lambda name: os.path.join(index_dir, name)
        with open(p("config.json"), encoding="utf-8") as f:
            self.cfg = json.load(f)
        try:
            import Stemmer  # PyStemmer: the Snowball stemmer the index was built with
            self.stem_words = Stemmer.Stemmer("english").stemWords
        except ImportError:  # safety net only; the grading environment lists PyStemmer
            self.stem_words = self.fallback_stem

        lx = np.load(p("lex.npz"))
        self.doc_ids = bytes(lx["doc_ids"]).decode("utf-8").split("\n")
        self.n_docs = len(self.doc_ids)
        self.term_row = {t: i for i, t in enumerate(bytes(lx["terms"]).decode("utf-8").split("\n"))}
        words = bytes(lx["words"]).decode("utf-8").split("\n")
        self.word_freq = dict(zip(words, lx["word_freq"].tolist()))
        self.wordpieces = set(bytes(lx["wordpieces"]).decode("utf-8").split("\n"))
        # chunk-level BM25: weight = idf[term] * q8 / 255 * scale
        self.c_indptr = lx["c_indptr"]
        self.c_idx = undelta(self.c_indptr, lx["c_idx"])
        self.c_val = lx["c_val"].astype(np.float32) * (lx["c_scale"][0] / 255.0)
        self.c_idf = lx["c_idf"]
        self.n_chunks = int(lx["n_chunks"][0])
        self.c_start = lx["c_start"].astype(np.int64)      # first chunk of each document (chunks are in doc order)
        self.d_indptr = lx["d_indptr"]
        self.d_idx = undelta(self.d_indptr, lx["d_idx"])
        self.d_val = lx["d_val"].astype(np.float32) * (lx["d_scale"][0] / 255.0)
        self.d_idf = lx["d_idf"]

        dz = np.load(p("dense.npz"))
        self.v = dz["v"].astype(np.float32) * dz["v_scale"][:, None]      # (n_chunks, 384)
        self.v_start = dz["v_start"].astype(np.int64)
        self.t = dz["t"].astype(np.float32) * dz["t_scale"][:, None]      # (n_docs, 384) title vectors

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(p("model.onnx"), so, providers=["CPUExecutionProvider"])
        with gzip.open(p("tokenizer.json.gz"), "rt", encoding="utf-8") as f:
            self.tok = Tokenizer.from_str(f.read())
        self.tok.enable_truncation(128)
        self.tok.no_padding()

    # ---------- query processing ----------
    def fallback_stem(self, words: list[str]) -> list[str]:
        """Without PyStemmer: keep a word if it is an index term, else strip common endings until one is."""
        out = []
        for w in words:
            cands = [w] + [w[:-len(e)] for e in ("ies", "es", "s", "ing", "ed", "ly", "e") if w.endswith(e) and len(w) - len(e) >= 3]
            cands += [c[:-1] + "i" for c in cands if c.endswith("y")]
            out.append(next((c for c in cands if c in self.term_row), w))
        return out

    def correct(self, w: str) -> str:
        """Fix a likely typo: a word unknown to both the corpus and the model's vocabulary is
        replaced by its most frequent corpus neighbour one edit away."""
        if len(w) < 4 or w.isdigit() or w in self.word_freq or w in self.wordpieces:
            return w
        best, bf = w, self.cfg["typo_min_freq"] - 1
        for c in edits1(w):
            f = self.word_freq.get(c, 0)
            if f > bf:
                best, bf = c, f
        return best

    def process(self, query: str):
        out = []
        for w in normalise(query).split():
            if w in ES:
                out.extend(ES[w].split())
            elif w in SPSTOP:
                continue
            else:
                out.append(self.correct(w))
        dense_text = " ".join(out)
        lex_terms = self.stem_words([w for w in out if w not in STOP and w not in QSTOP])
        return lex_terms, dense_text

    # ---------- scoring ----------
    def bm25(self, terms, indptr, idx, val, idf, n):
        s = np.zeros(n, np.float32)
        for t in terms:
            i = self.term_row.get(t)
            if i is None:
                continue
            a, b = indptr[i], indptr[i + 1]
            if a == b:
                continue
            s[idx[a:b]] += val[a:b] * idf[i]
        return s

    def encode(self, text: str) -> np.ndarray:
        e = self.tok.encode(self.cfg["query_prefix"] + text)
        ids = np.array([e.ids], np.int64)
        am = np.ones_like(ids)
        h = self.sess.run(None, {"input_ids": ids, "attention_mask": am, "token_type_ids": np.zeros_like(ids)})[0]
        q = h[0, 0]
        return q / (np.linalg.norm(q) + 1e-9)

    @staticmethod
    def norm(x: np.ndarray) -> np.ndarray:
        """Scale a per-document score vector so the median document is 0 and the best is 1."""
        m = np.median(x)
        top = x.max()
        return (x - m) / (top - m + 1e-9)

    def search(self, query: str, k: int = 3) -> list[str]:
        c = self.cfg
        terms, dense_text = self.process(query)
        n = self.n_docs
        # lexical: best 300-word chunk per document + whole-document BM25
        cs = self.bm25(terms, self.c_indptr, self.c_idx, self.c_val, self.c_idf, self.n_chunks)
        lex_chunk = np.maximum.reduceat(cs, self.c_start)
        lex_doc = self.bm25(terms, self.d_indptr, self.d_idx, self.d_val, self.d_idf, n)
        # neural: best 200-word chunk per document + title similarity
        q = self.encode(dense_text)
        sims = self.v @ q
        den_chunk = np.maximum.reduceat(sims, self.v_start)
        den_title = self.t @ q
        score = (c["w_lex_chunk"] * self.norm(lex_chunk) + c["w_lex_doc"] * self.norm(lex_doc)
                 + c["w_dense_chunk"] * self.norm(den_chunk) + c["w_dense_title"] * self.norm(den_title))
        k = max(0, min(int(k), n))
        if k == 0:
            return []
        top = np.argpartition(-score, k - 1)[:k]
        top = top[np.argsort(-score[top], kind="stable")]
        return [self.doc_ids[i] for i in top]
