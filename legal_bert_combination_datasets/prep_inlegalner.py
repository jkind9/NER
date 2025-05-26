#!/usr/bin/env python
"""
prep_inlegalner.py  –  convert span-labelled InLegalNER to token-level BIO
Outputs ./inlegalner_tokenized/{train,dev,test}.arrow
"""

from pathlib import Path
import spacy
from datasets import load_dataset, DatasetDict, Features, Sequence, ClassLabel, Value
def span_bounds(span):
    """
    Return (start, end, label) or None for unsupported/empty span.
    Handles three known LS formats; returns None if coordinates missing.
    """
    # Style A ─ direct numeric coordinates
    if "start" in span and "end" in span:
        lab = span.get("label") or span.get("labels")
        lab = lab[0] if isinstance(lab, list) else lab
        return span["start"], span["end"], lab

    # Style B ─ points wrapper
    if span.get("points"):
        pt = span["points"][0]
        lab = span.get("label") or span.get("labels")
        lab = lab[0] if isinstance(lab, list) else lab
        return pt["start"], pt["end"], lab

    # Style C — Label-Studio “result” list
    if span.get("result"):
        for res in span["result"]:
            if res.get("type") == "labels" and res.get("value"):
                v = res["value"]
                yield v["start"], v["end"], v["labels"][0]
        return  # handled—all yields done
    # Otherwise → no usable coords
    return None


def spans_to_bio(example):
    """
    Convert every annotated span in `example` to BIO tags.

    * Handles any number of fragments inside each `span["result"]`.
    * Ignores spans with no usable coordinates.
    """
    text = example["data"]["text"]
    doc  = nlp(text)
    tags = [0] * len(doc)                # all “O”

    for span in example["annotations"]:
        # span_bounds now *yields* 0-N (start, end, label) tuples
        for bounds in span_bounds(span) or []:
            s, e, lab = bounds
            for tok in doc:
                if tok.idx >= s and tok.idx + len(tok) <= e:
                    prefix = "B" if tok.idx == s else "I"
                    tags[tok.i] = lbl2id[f"{prefix}-{lab}"]
    return {
        "tokens": [t.text for t in doc],
        "ner_tags": tags,
    }



# ---- spaCy tokenizer -------------------------------------------------------
nlp = spacy.blank("en")

# ---- load gated dataset ----------------------------------------------------
raw = load_dataset("opennyaiorg/InLegalNER")          # train/dev/test
LABELS = [
    "LAWYER","COURT","JUDGE","PETITIONER","RESPONDENT",
    "CASE_NUMBER","GPE","DATE","ORG","STATUTE","WITNESS",
    "PRECEDENT","PROVISION","OTHER_PERSON",
]
label_names = ["O"] + [f"{p}-{l}" for l in LABELS for p in ("B","I")]
lbl2id = {t:i for i,t in enumerate(label_names)}

print("Converting spans → BIO …")
conv = DatasetDict()
for split in raw:
    conv[split] = raw[split].map(
        spans_to_bio,
        remove_columns=raw[split].column_names,
        desc=f"processing {split}"
    )



# ---- cast schema -----------------------------------------------------------
out_root = Path(r"C:\Users\jkind\Documents\NER\local_data/inlegalner_tokenized")   # or whatever path you prefer
out_root.mkdir(parents=True, exist_ok=True)

for split in ("train", "dev", "test"):
    file = out_root / f"{split}.arrow"
    conv[split].save_to_disk(file) 