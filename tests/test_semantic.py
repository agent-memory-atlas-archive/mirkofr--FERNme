"""Optional meaning search (fernme.semantic). Runs without the model: a fake embedder
stands in, and model downloads are served from a local folder. Fictional data only."""
import hashlib
import io
import os

import numpy as np
import pytest

from fernme import semantic


def fake_embed(vocab):
    """Each text becomes a bag-of-concepts vector, so different words can share meaning."""
    concepts = sorted({c for cs in vocab.values() for c in cs})

    def embed(texts):
        out = []
        for t in texts:
            v = np.zeros(len(concepts))
            for word, cs in vocab.items():
                if word in t.lower():
                    for c in cs:
                        v[concepts.index(c)] += 1
            n = np.linalg.norm(v)
            out.append(v / n if n else v)
        return np.array(out)
    return embed


VOCAB = {"home": ["home"], "house": ["home"], "дом": ["home"], "museum": ["museum"],
         "tea": ["drink"], "coffee": ["drink"]}


def test_similarity_finds_meaning_across_words_and_languages():
    idx = semantic.TextIndex(fake_embed(VOCAB), floor=0.25)
    notes = ["Sam's home is near Busan Museum.", "Sam prefers tea in the morning."]
    sims = idx.similarities("Где мой дом?", notes)
    assert sims[0] > 0.5 > sims[1]
    sims = idx.similarities("coffee or not?", notes)
    assert sims[1] > sims[0]


def test_text_vectors_are_cached_by_content():
    calls = []
    base = fake_embed(VOCAB)

    def counting(texts):
        calls.append(list(texts))
        return base(texts)
    idx = semantic.TextIndex(counting, floor=0.25)
    notes = ["home one", "tea two", "home one"]
    idx.similarities("house", notes)
    idx.similarities("house", notes)
    embedded_notes = [t for c in calls for t in c if t != "house"]
    assert sorted(embedded_notes) == ["home one", "tea two"]  # each note once, duplicates folded


def test_hybrid_keeps_keyword_hits_and_drops_unrelated():
    assert semantic.hybrid(0.0, 0.10, floor=0.25) is None
    assert semantic.hybrid(0.4, 0.10, floor=0.25) == pytest.approx(0.2)
    assert semantic.hybrid(0.0, 0.625, floor=0.25) == pytest.approx(0.25)
    strong = semantic.hybrid(0.8, 0.9, floor=0.25)
    weak = semantic.hybrid(0.0, 0.9, floor=0.25)
    assert strong > weak


def test_load_returns_none_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv(semantic.MODEL_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(semantic, "_loaded", {})
    assert semantic.missing()
    assert semantic.load() is None


def _serve(tmp_path, monkeypatch, payloads):
    """Pretend model repo under a file:// URL with the given file contents."""
    spec = dict(semantic.MODELS["minilm-int8"])
    files = {}
    for local, data in payloads.items():
        remote = f"files/{local}"
        src = tmp_path / "repo" / spec["repo"] / "resolve" / spec["revision"] / remote
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(data)
        files[local] = (remote, len(data), hashlib.sha256(data).hexdigest())
    spec["files"] = files
    monkeypatch.setitem(semantic.MODELS, "test-model", spec)
    return (tmp_path / "repo").as_uri()


def test_fetch_verifies_and_installs(tmp_path, monkeypatch):
    base = _serve(tmp_path, monkeypatch, {"model.onnx": b"weights", "tokenizer.json": b"{}"})
    dest = semantic.fetch("test-model", tmp_path / "models", base_url=base, out=io.StringIO())
    assert (dest / "model.onnx").read_bytes() == b"weights"
    log = io.StringIO()
    semantic.fetch("test-model", dest, base_url=base, out=log)  # second run: nothing downloaded
    assert "download" not in log.getvalue()


def test_fetch_rejects_a_changed_file(tmp_path, monkeypatch):
    base = _serve(tmp_path, monkeypatch, {"model.onnx": b"weights", "tokenizer.json": b"{}"})
    files = dict(semantic.MODELS["test-model"]["files"])
    remote, size, _sha = files["model.onnx"]
    files["model.onnx"] = (remote, size, "0" * 64)
    semantic.MODELS["test-model"]["files"] = files
    with pytest.raises(ValueError):
        semantic.fetch("test-model", tmp_path / "models", base_url=base, out=io.StringIO())
    assert not (tmp_path / "models" / "model.onnx").exists()
    assert not list((tmp_path / "models").glob("*.part"))


@pytest.mark.skipif(bool(semantic.missing()), reason="meaning search not installed (fernme[semantic] + fernme-model fetch)")
def test_real_model_matches_across_languages():
    idx = semantic.load()
    notes = ["Sam's home address is 1 Example Street.", "Sam likes green tea.", "Sam's daughter was born in 2015."]
    for q in ("Где мой дом?", "우리 집 주소가 뭐야?", "Where do I live?"):
        sims = idx.similarities(q, notes)
        assert sims.index(max(sims)) == 0, q
