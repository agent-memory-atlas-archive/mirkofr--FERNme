# Meaning search (optional)

FERNme's write and recall core matches tags and words. A question worded differently
from the note ("where do I live?" vs "home location is ...") or asked in another
language ("Где мой дом?") can then miss it. `fernme.semantic` adds local meaning
search for such cases: a small multilingual model turns texts into vectors on the
CPU. Nothing leaves the machine and the write path is unchanged.

## Install

```bash
pip install "fernme[semantic]"   # onnxruntime + tokenizers
fernme-model fetch               # ~128 MB, pinned revision, SHA-256 checked
fernme-model status              # says what is missing, if anything
```

Files go to `~/.fernme/models/<model>` (or `$FERNME_MODEL_DIR/<model>`).

## Use

```python
from fernme import semantic

index = semantic.load()          # None when the extra or the model is missing
if index is None:
    ...                          # keep keyword search
sims = index.similarities("Где мой дом?", note_texts)
score = semantic.hybrid(keyword_score, sims[i], index.floor)   # None = unrelated
```

`hybrid()` blends a keyword score (0..1) with the cosine similarity, so exact
matches keep winning, and returns `None` when neither says the note is related.
Note vectors are cached by content, so only new or edited notes are embedded.

## Model and cost

`minilm-int8`: paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0, 50+ languages),
int8 ONNX export. Measured on a 6-core desktop CPU with 2 threads: ~420 MB RAM once
loaded, ~3 ms per query, ~20 ms per new note. It reads the first ~128 tokens of a
note. `FERNME_SEMANTIC_THREADS` sets the CPU threads (default 2).

On a 96-question test over a profile of 64 notes, with questions in four languages,
keyword recall put the right note in the top 3 for 36% of the questions. Hybrid
search did so for 76%. For questions in another language than the notes it went
from near 0% to 53-78%. Questions with no matching memory still return nothing
most of the time (floor 0.25).
