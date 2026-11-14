#!/usr/bin/env python3
"""
LMML 2026 - task_usa: "The Great American Token Mystery"

Reads  test-input.json  from the current working directory and writes
test-output.json  into the same directory:

    {"tokenizer_id": "...", "decoded_message": "..."}

Optional: python main.py <input.json> <output.json>   (e.g. for the training sample)

How it works
1. Candidate tokenizers: every OpenAI tiktoken encoding (r50k_base / gpt2,
   p50k_base, p50k_edit, cl100k_base, o200k_base, o200k_harmony).
   Their vocabulary files are bundled in ./vocab (SHA-256 checked against the
   official tiktoken hashes), so no internet connection is needed. If ./vocab is
   missing, the official tiktoken.get_encoding() is used instead.
2. For each candidate we score the calibration pair:
      score 2 = encode(calibration.text) == calibration.token_ids   (exact match)
      score 1 = decode(calibration.token_ids) == calibration.text     (round-trip only)
   The candidate with the highest score wins (ties -> earlier in CANDIDATES).
3. The winner decodes challenge.token_ids -> decoded_message.
4. Only if no tiktoken encoding matches at all, a fallback tries common
   Hugging Face tokenizers (needs `transformers` + internet).
"""
import base64
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VOCAB_DIR = HERE / "vocab"

ENDOFTEXT = "<|endoftext|>"
FIM_PREFIX = "<|fim_prefix|>"
FIM_MIDDLE = "<|fim_middle|>"
FIM_SUFFIX = "<|fim_suffix|>"
ENDOFPROMPT = "<|endofprompt|>"

# Regexes copied from tiktoken_ext/openai_public.py (tiktoken 0.14.0)
R50K_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
CL100K_PAT = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
O200K_PAT = "|".join([
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""\p{N}{1,3}""",
    r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
    r"""\s*[\r\n]+""",
    r"""\s+(?!\S)""",
    r"""\s+""",
])

VOCAB_FILES = {  # file name -> official SHA-256 from tiktoken
    "r50k_base.tiktoken": "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930",
    "p50k_base.tiktoken": "94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069",
    "cl100k_base.tiktoken": "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
    "o200k_base.tiktoken": "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
}

HARMONY_SPECIALS = {
    "<|startoftext|>": 199998, ENDOFTEXT: 199999, "<|reserved_200000|>": 200000,
    "<|reserved_200001|>": 200001, "<|return|>": 200002, "<|constrain|>": 200003,
    "<|reserved_200004|>": 200004, "<|channel|>": 200005, "<|start|>": 200006,
    "<|end|>": 200007, "<|message|>": 200008, "<|reserved_200009|>": 200009,
    "<|reserved_200010|>": 200010, "<|reserved_200011|>": 200011, "<|call|>": 200012,
}
HARMONY_SPECIALS.update({f"<|reserved_{i}|>": i for i in range(200013, 201088)})

# (tokenizer_id, vocab file, regex, special tokens) - order = tie-break priority
CANDIDATES = [
    ("p50k_base", "p50k_base.tiktoken", R50K_PAT, {ENDOFTEXT: 50256}),
    ("r50k_base", "r50k_base.tiktoken", R50K_PAT, {ENDOFTEXT: 50256}),  # == "gpt2"
    ("p50k_edit", "p50k_base.tiktoken", R50K_PAT,
     {ENDOFTEXT: 50256, FIM_PREFIX: 50281, FIM_MIDDLE: 50282, FIM_SUFFIX: 50283}),
    ("cl100k_base", "cl100k_base.tiktoken", CL100K_PAT,
     {ENDOFTEXT: 100257, FIM_PREFIX: 100258, FIM_MIDDLE: 100259, FIM_SUFFIX: 100260,
      ENDOFPROMPT: 100276}),
    ("o200k_base", "o200k_base.tiktoken", O200K_PAT, {ENDOFTEXT: 199999, ENDOFPROMPT: 200018}),
    ("o200k_harmony", "o200k_base.tiktoken", O200K_PAT,
     {**HARMONY_SPECIALS, ENDOFPROMPT: 200018}),
]

# Fallback only (needs transformers + internet): common HF tokenizers
HF_FALLBACK = [
    "gpt2", "bert-base-uncased", "bert-base-cased", "bert-base-multilingual-cased",
    "roberta-base", "distilbert-base-uncased", "albert-base-v2", "xlnet-base-cased",
    "t5-small", "google/mt5-small", "facebook/bart-base", "xlm-roberta-base",
    "facebook/opt-125m", "EleutherAI/gpt-neox-20b", "EleutherAI/gpt-j-6b",
    "bigscience/bloom-560m", "microsoft/phi-2", "Salesforce/codegen-350M-mono",
    "hf-internal-testing/llama-tokenizer", "mistralai/Mistral-7B-v0.1",
    "Qwen/Qwen2-0.5B", "tiiuae/falcon-7b", "google/flan-t5-small",
    "Xenova/gpt-4", "Xenova/gpt-4o", "Xenova/text-davinci-003",
]

_RANK_CACHE = {}


def load_ranks(fname):
    """Read a .tiktoken file (lines: base64(token_bytes) rank) -> {bytes: rank}."""
    if fname in _RANK_CACHE:
        return _RANK_CACHE[fname]
    raw = (VOCAB_DIR / fname).read_bytes()
    want = VOCAB_FILES.get(fname)
    if want and hashlib.sha256(raw).hexdigest() != want:
        raise ValueError(f"hash mismatch for {fname}")
    ranks = {}
    for line in raw.splitlines():
        if line:
            tok, rank = line.split()
            ranks[base64.b64decode(tok)] = int(rank)
    _RANK_CACHE[fname] = ranks
    return ranks


class PureDecoder:
    """Dependency-free decoder (used to verify round trips / when tiktoken is missing)."""

    def __init__(self, ranks, specials):
        self.id2bytes = {r: b for b, r in ranks.items()}
        for s, i in specials.items():
            self.id2bytes[i] = s.encode("utf-8")

    def decode(self, ids):
        out = []
        for i in ids:
            if i not in self.id2bytes:
                raise KeyError(i)
            out.append(self.id2bytes[i])
        return b"".join(out).decode("utf-8", errors="replace")


def build(name, fname, pat, specials):
    """Return (encode_fn or None, decode_fn) for a candidate."""
    if not (VOCAB_DIR / fname).exists():
        # vocab folder not shipped next to main.py -> official tiktoken (downloads/caches)
        import tiktoken
        enc = tiktoken.get_encoding(name)
        return (lambda t: enc.encode(t, allowed_special=set(), disallowed_special=())), enc.decode
    ranks = load_ranks(fname)
    pure = PureDecoder(ranks, specials)
    try:
        import tiktoken
        enc = tiktoken.Encoding(name=name, pat_str=pat, mergeable_ranks=ranks,
                                special_tokens=specials)
        return (lambda t: enc.encode(t, allowed_special=set(), disallowed_special=())), pure.decode
    except Exception:
        return None, pure.decode


def norm(s):
    return s.replace("\r\n", "\n").replace("\r", "\n").rstrip()


def identify_tiktoken(text, ids, log):
    best = None  # (score, priority, name, decode_fn)
    for prio, (name, fname, pat, specials) in enumerate(CANDIDATES):
        try:
            encode, decode = build(name, fname, pat, specials)
        except Exception as e:
            log(f"  {name:14s} unavailable ({e})")
            continue
        score = 0
        enc_ok = None
        if encode is not None:
            try:
                enc_ok = encode(text) == ids
            except Exception:
                enc_ok = False
        try:
            dec_ok = decode(ids) == text
        except Exception:
            dec_ok = False
        if enc_ok:
            score = 2
        elif dec_ok:
            score = 1
        log(f"  {name:14s} encode_match={enc_ok} decode_roundtrip={dec_ok} score={score}")
        if score and (best is None or score > best[0]):
            best = (score, prio, name, decode)
    return best


def identify_hf(text, ids, log):
    try:
        from transformers import AutoTokenizer
    except Exception:
        log("  transformers not installed - HF fallback skipped")
        return None
    best = None
    for mid in HF_FALLBACK:
        try:
            tok = AutoTokenizer.from_pretrained(mid)
        except Exception as e:
            log(f"  {mid}: cannot load ({type(e).__name__})")
            continue
        for add_sp in (False, True):
            try:
                if tok.encode(text, add_special_tokens=add_sp) == ids:
                    log(f"  {mid}: exact match (add_special_tokens={add_sp})")
                    return mid, tok
            except Exception:
                pass
        try:
            d = tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if norm(d) == norm(text) and best is None:
                best = (mid, tok)
        except Exception:
            pass
    return best


def solve(inp_path, out_path, verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    data = json.loads(Path(inp_path).read_text(encoding="utf-8"))
    text = data["calibration"]["text"]
    cal_ids = [int(x) for x in data["calibration"]["token_ids"]]
    ch_ids = [int(x) for x in data["challenge"]["token_ids"]]

    log(f"[{inp_path}] calibration: {len(cal_ids)} ids, challenge: {len(ch_ids)} ids")
    best = identify_tiktoken(text, cal_ids, log)
    if best is not None:
        tok_id, message = best[2], best[3](ch_ids)
    else:
        log("No tiktoken encoding matched - trying Hugging Face fallback")
        hf = identify_hf(text, cal_ids, log)
        if hf is None:
            raise SystemExit("No candidate tokenizer reproduces the calibration pair.")
        tok_id, tok = hf
        message = tok.decode(ch_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    result = {"tokenizer_id": tok_id, "decoded_message": message}
    Path(out_path).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8")
    log(f"=> tokenizer_id: {tok_id}")
    log(f"=> decoded_message: {message!r}")
    log(f"written to {out_path}")
    return result


if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else "test-input.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "test-output.json"
    solve(inp, out)
