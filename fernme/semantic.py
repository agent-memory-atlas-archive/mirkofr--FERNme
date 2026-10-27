"""Optional meaning search: local multilingual sentence embeddings on the CPU.

Keyword matching misses a memory when a question uses other words or another
language than the note ("Где мой дом?" vs "home location is ..."). This module
turns texts into meaning vectors with a small local model so such questions still
find the note. Nothing leaves the machine and no model call is made at write time.

It is off unless installed:

    pip install "fernme[semantic]"     # onnxruntime + tokenizers
    fernme-model fetch                 # ~128 MB model, checked against pinned SHA-256s

Callers treat it as an extra: ``load()`` returns ``None`` when the packages or the
model files are missing, and the caller keeps its keyword search. ``hybrid()``
blends a keyword score with a meaning score so exact matches keep winning.

Model: paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0, 50+ languages), int8
ONNX export, ~420 MB RAM once loaded, ~3 ms per query on a 6-core desktop CPU.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import threading
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

MODELS: Dict[str, dict] = {
    "minilm-int8": {
        "repo": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "revision": "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
        "license": "Apache-2.0",
        "files": {
            # local name: (path in the repo, size in bytes, sha256)
            "model.onnx": ("onnx/model_quint8_avx2.onnx", 118453870,
                           "98a01d88b7de996cdea58c32ca71208c09968d143798814b2ea09d3439dc334f"),
            "tokenizer.json": ("tokenizer.json", 9081518,
                               "2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8"),
        },
        "max_tokens": 128,
        # Below this cosine a note is treated as unrelated (unless a keyword matched).
        "floor": 0.25,
    },
}
DEFAULT_MODEL = "minilm-int8"
BASE_URL = "https://huggingface.co"
MODEL_DIR_ENV = "FERNME_MODEL_DIR"
THREADS_ENV = "FERNME_SEMANTIC_THREADS"


def model_root() -> Path:
    return Path(os.environ.get(MODEL_DIR_ENV) or Path.home() / ".fernme" / "models")


def model_dir(name: str = DEFAULT_MODEL) -> Path:
    return model_root() / name


def missing(name: str = DEFAULT_MODEL) -> List[str]:
    """What stops meaning search from loading; empty when it can."""
    out = []
    for mod in ("numpy", "onnxruntime", "tokenizers"):
        try:
            __import__(mod)
        except ImportError:
            out.append(f'package {mod} (pip install "fernme[semantic]")')
    d = model_dir(name)
    for local, (_path, size, _sha) in MODELS[name]["files"].items():
        f = d / local
        if not f.is_file() or f.stat().st_size != size:
            out.append(f"model file {f} (run: fernme-model fetch)")
    return out


class Embedder:
    """Mean-pooled, L2-normalised sentence vectors from an ONNX transformer."""

    def __init__(self, name: str = DEFAULT_MODEL, threads: Optional[int] = None):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._np = np
        spec = MODELS[name]
        d = model_dir(name)
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_truncation(spec["max_tokens"])
        self.tok.enable_padding()
        so = ort.SessionOptions()
        # Few threads: memory recall shares the machine with the agents it serves.
        so.intra_op_num_threads = int(threads or os.environ.get(THREADS_ENV) or 2)
        self.sess = ort.InferenceSession(str(d / "model.onnx"), so, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def embed(self, texts: Sequence[str]):
        np = self._np
        enc = self.tok.encode_batch(list(texts))
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = self.sess.run(None, feed)[0]
        m = mask[..., None].astype(np.float32)
        vec = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
        return vec / np.clip(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9, None)


class TextIndex:
    """Cosine similarity of a query to many texts; text vectors are cached by content hash."""

    BATCH = 8
    MAX_CACHE = 20_000

    def __init__(self, embed: Callable[[Sequence[str]], "object"], floor: float):
        import numpy as np

        self._np = np
        self._embed = embed
        self.floor = floor
        self._cache: Dict[str, "object"] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def vectors(self, texts: Sequence[str]):
        np = self._np
        keys = [self._key(t) for t in texts]
        with self._lock:
            todo = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in self._cache]
            for i in range(0, len(todo), self.BATCH):
                part = todo[i:i + self.BATCH]
                for (k, _t), v in zip(part, self._embed([t for _k, t in part])):
                    self._cache[k] = v
            if len(self._cache) > self.MAX_CACHE:  # notes that were edited or forgotten
                live = set(keys)
                self._cache = {k: v for k, v in self._cache.items() if k in live}
            return np.vstack([self._cache[k] for k in keys]) if keys else np.zeros((0, 1))

    def similarities(self, query: str, texts: Sequence[str]) -> List[float]:
        if not texts or not query.strip():
            return [0.0] * len(texts)
        q = self._embed([query])[0]
        return [float(x) for x in self.vectors(texts) @ q]


def hybrid(keyword: float, similarity: float, floor: float, alpha: float = 0.5) -> Optional[float]:
    """Blend a keyword score (0..1) with a cosine similarity.

    Returns ``None`` when neither signal says the text is related, so callers can
    leave it out instead of padding results with unrelated notes.
    """
    if keyword <= 0 and similarity < floor:
        return None
    meaning = max(0.0, (similarity - floor) / (1.0 - floor))
    return alpha * keyword + (1.0 - alpha) * meaning


_loaded: Dict[str, Optional[TextIndex]] = {}
_load_lock = threading.Lock()


def load(name: str = DEFAULT_MODEL) -> Optional[TextIndex]:
    """The shared index for ``name``, or ``None`` when meaning search is not installed."""
    with _load_lock:
        if name not in _loaded:
            if missing(name):
                _loaded[name] = None
            else:
                emb = Embedder(name)
                _loaded[name] = TextIndex(emb.embed, MODELS[name]["floor"])
        return _loaded[name]


def fetch(name: str = DEFAULT_MODEL, dest: Optional[Path] = None, base_url: str = BASE_URL,
          out=sys.stdout) -> Path:
    """Download the model files at the pinned revision and verify size and SHA-256."""
    spec = MODELS[name]
    d = Path(dest) if dest else model_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    for local, (path, size, sha) in spec["files"].items():
        target = d / local
        if target.is_file() and target.stat().st_size == size and _sha256(target) == sha:
            print(f"ok       {local}", file=out)
            continue
        url = f"{base_url}/{spec['repo']}/resolve/{spec['revision']}/{path}"
        part = target.with_name(target.name + ".part")
        print(f"download {local} ({size / 1e6:.0f} MB) from {url}", file=out)
        h = hashlib.sha256()
        with urllib.request.urlopen(url) as r, open(part, "wb") as f:  # noqa: S310 (pinned https URL)
            for chunk in iter(lambda: r.read(1 << 20), b""):
                h.update(chunk)
                f.write(chunk)
        if part.stat().st_size != size or h.hexdigest() != sha:
            part.unlink()
            raise ValueError(f"{local}: size or SHA-256 does not match the pinned file; not installed")
        part.replace(target)
    return d


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="fernme-model", description="Local model for FERNme meaning search.")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download the model (pinned revision, SHA-256 checked)")
    f.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS))
    f.add_argument("--dir", help=f"models folder (default ${MODEL_DIR_ENV} or ~/.fernme/models); set {MODEL_DIR_ENV} to the same folder")
    s = sub.add_parser("status", help="say whether meaning search can load")
    s.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS))
    a = p.parse_args(argv)
    if a.cmd == "fetch":
        spec = MODELS[a.model]
        print(f"{spec['repo']} @ {spec['revision'][:12]} ({spec['license']})")
        print(f"installed in {fetch(a.model, Path(a.dir) / a.model if a.dir else None)}")
        return 0
    gaps = missing(a.model)
    if not gaps:
        print(f"meaning search ready: {model_dir(a.model)}")
        return 0
    print("meaning search off; missing:\n  " + "\n  ".join(gaps))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
