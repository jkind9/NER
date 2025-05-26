#!/usr/bin/env python
"""train_legal_ner_union.py – v5  (May 2025)
Final stabilisation: handles *nested* “data” structures such as those in
`opennyaiorg/InLegalNER`’s raw dump.

What changed
────────────
1. **`flatten_if_needed()`** – if a split contains a single `data` column with a
   dict per row, it extracts `tokens` & `ner_tags` (or `tags`) and drops the
   wrapper.
2. Tag‑column detection now runs *after* flattening.
3. Error messages show actual columns so you can debug new corpora easily.

No other logic touched; still early‑stopping, loss PNG, held‑out test set.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score
from datasets import load_from_disk
from datasets import DatasetDict, Sequence, ClassLabel, Features, Value
import torch
import matplotlib.pyplot as plt
from datasets import (
    load_dataset, Dataset, DatasetDict, concatenate_datasets,
    ClassLabel, Sequence
)
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, TrainingArguments,
    DataCollatorForTokenClassification, Trainer, EarlyStoppingCallback, pipeline,
    BitsAndBytesConfig
)


try:
    from peft import LoraConfig, get_peft_model
    PEFT = True
except ImportError:
    PEFT = False

LABEL_ALL_TOKENS = False

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def flatten_if_needed(ds: DatasetDict) -> DatasetDict:
    """If dataset rows are wrapped in a `data` dict, unwrap them."""
    if "data" not in ds["train"].column_names:
        return ds

    def _flat(ex):
        inner = ex["data"]
        return {
            "tokens": inner.get("tokens") or inner.get("words"),
            "ner_tags": inner.get("ner_tags") or inner.get("tags") or inner.get("labels"),
        }
    cols_to_remove = ds["train"].column_names  # keep only new keys
    return ds.map(_flat, remove_columns=cols_to_remove)


def detect_tag_col(ds_split):
    """Return the first column that looks like a NER tag sequence."""
    candidates = ("ner_tags", "tags", "coarse_grained", "fine_grained")
    for c in candidates:
        if c in ds_split.column_names:
            return c
    raise ValueError(f"No tag column found. Columns present: {ds_split.column_names}")



def ensure_int_tags(ds: DatasetDict, tag_col: str) -> DatasetDict:
    """
    • Renames the given tag column to `ner_tags`.
    • Guarantees tags are integers backed by ClassLabel.
    • Removes the original tag column from the feature dict before casting.
    """
    first_val = ds["train"][tag_col][0][0]          # peek at first tag
    tags_are_int = isinstance(first_val, (int, bool))

    # ------ 1. Build ClassLabel ------------------------------------------------
    if tags_are_int:
        max_id = max(max(row) for row in ds["train"][tag_col])
        cl = ClassLabel(num_classes=max_id + 1)
        convert = False
    else:
        vocab = sorted({t for row in ds["train"][tag_col] for t in row})
        cl = ClassLabel(names=vocab)
        convert = True

    # ------ 2. Optionally convert strings → ints ------------------------------
    if convert:
        def to_int(ex):
            ex["ner_tags"] = [cl.str2int(t) for t in ex[tag_col]]
            return ex
        ds = ds.map(to_int, remove_columns=[tag_col])
    else:
        if tag_col != "ner_tags":
            ds = ds.rename_column(tag_col, "ner_tags")

    # ------ 3. Re-create Features *after* column removal ----------------------
    new_features = Features(
        {
            **{k: v for k, v in ds["train"].features.items() if k != "ner_tags"},
            "ner_tags": Sequence(cl),
        }
    )

    return ds.cast(new_features)


def tokenize_and_align(ex, tok, max_len):
    enc = tok(ex["tokens"], truncation=True, is_split_into_words=True, max_length=max_len)
    labels = []
    for i, wids in enumerate(enc.word_ids(batch_index=i) for i in range(len(enc["input_ids"]))):
        wl = ex["ner_tags"][i]; prev = None; row = []
        for wid in wids:
            if wid is None: row.append(-100)
            elif wid != prev: row.append(wl[wid])
            else: row.append(wl[wid] if LABEL_ALL_TOKENS else -100)
            prev = wid
        labels.append(row)
    enc["labels"] = labels
    return enc


def label_list(ds: DatasetDict) -> List[str]:
    return ds["train"].features["ner_tags"].feature.names

# ─────────────────────────────────────────────────────────────────────────────
# Dataset union builder
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(sample_finer: int | None = None, seed: int = 42, pct_test: float = 0.05):
    base = "C:/Users/jkind/Documents/NER/local_data/inlegalner_tokenized"

    ds_in = load_dataset(
        "arrow",
        data_files={
            "train": f"{base}/train.arrow/*.arrow",
            "validation": f"{base}/dev.arrow/*.arrow",
            "test": f"{base}/test.arrow/*.arrow",
        },
    )
    ds_map = flatten_if_needed(load_dataset("joelniklaus/mapa").filter(lambda e: e.get("language", "en") == "en"))
    ds_fin = flatten_if_needed(load_dataset("nlpaueb/finer-139"))

    if sample_finer:
        ds_fin["train"] = ds_fin["train"].shuffle(seed=seed).select(range(sample_finer))

    ds_in  = ensure_int_tags(ds_in,  detect_tag_col(ds_in["train"]))
    ds_map = ensure_int_tags(ds_map, detect_tag_col(ds_map["train"]))
    ds_fin = ensure_int_tags(ds_fin, detect_tag_col(ds_fin["train"]))

    labels = sorted(set(label_list(ds_in)) | set(label_list(ds_map)) | set(label_list(ds_fin)))
    id_map = {l: i for i, l in enumerate(labels)}

    def _remap(ds_dict: DatasetDict, labels: list[str]) -> DatasetDict:
        """
        Take a DatasetDict with a 'ner_tags' column of ints,
        and remap every tag to the new union label space.
        Returns a brand-new DatasetDict (never a plain dict).
        """
        # build old_id -> new_id mapping
        src_names = ds_dict["train"].features["ner_tags"].feature.names
        name2new  = {name: i for i, name in enumerate(labels)}
        mapping   = {old_id: name2new[src_names[old_id]] for old_id in range(len(src_names))}

        # feature schema for the remapped ner_tags
        new_feats = ds_dict["train"].features.copy()
        new_feats["ner_tags"] = Sequence(ClassLabel(names=labels))

        def convert(ex):
            ex["ner_tags"] = [mapping[i] for i in ex["ner_tags"]]
            return ex

        remapped_splits = {}
        for split_name, ds in ds_dict.items():
            # ds is a Dataset; map it *with* the new schema
            remapped_splits[split_name] = ds.map(
                convert,
                features=new_feats,
                batched=False,    # your tags are per-example
            )

        return DatasetDict(remapped_splits)
    
    ds_in  = _remap(ds_in, labels)
    ds_map = _remap(ds_map, labels)
    ds_fin = _remap(ds_fin, labels)

    def up(split, tgt: int = 100_000):
        """
        Ensure `split` has at least `tgt` examples by up-sampling.
        Always returns a `Dataset`, never a plain dict.
        """
        n = len(split)
        if n >= tgt:
            return split
        # calculate how many times to repeat the dataset
        reps = -(-tgt // n)  # ceiling division
        concatenated = concatenate_datasets([split] * reps)
        # select exactly `tgt` examples, using `.select()` to return a Dataset
        return concatenated.select(range(tgt))
    
    def balance_split(split, tgt: int = 100_000, seed: int = 42):
        """
        Return a Dataset of exactly `tgt` examples:
        • If the input is larger → down-sample
        • If smaller        → up-sample via `up()`
        """
        n = len(split)
        if n > tgt:
            return split.shuffle(seed=seed).select(range(tgt))
        return up(split, tgt)

    
    target = 10_000
    in_bal  = balance_split(ds_in["train"],  target, seed)
    map_bal = balance_split(ds_map["train"], target, seed)
    fin_bal = balance_split(ds_fin["train"], target, seed)

    pool = concatenate_datasets([in_bal, map_bal, fin_bal]).shuffle(seed=seed)

    interim = pool.train_test_split(test_size=pct_test, seed=seed)
    tmp = interim["test"].train_test_split(test_size=0.5, seed=seed)
    return DatasetDict(train=interim["train"], validation=tmp["train"], test=tmp["test"]), labels

# ─────────────────────────────────────────────────────────────────────────────
# Plot helper
# ─────────────────────────────────────────────────────────────────────────────

def plot_history(trainer, out_png: Path):
    tr_e, tr_l, va_l = [], [], []
    for rec in trainer.state.log_history:
        if "loss" in rec and "epoch" in rec and "eval_loss" not in rec:
            tr_e.append(rec["epoch"]); tr_l.append(rec["loss"])
        if "eval_loss" in rec:
            va_l.append(rec["eval_loss"])
    if tr_e:
        plt.figure(figsize=(6,4))
        plt.plot(tr_e, tr_l, label="train")
        if va_l: plt.plot(tr_e[:len(va_l)], va_l, label="val")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
        plt.savefig(out_png, dpi=300); plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
 # ─── 1) Arguments ─────────────────────────────────────────
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir",                        default="./eng_legal_ner_union")
    p.add_argument("--num_train_epochs",       type=int, default=10)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate",         type=float, default=2e-5)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size",  type=int, default=4)
    p.add_argument("--max_length",                    type=int, default=256)
    p.add_argument("--sample_finer",          type=int, default=0)
    p.add_argument(
        "--no_lora",
        dest="use_lora",
        action="store_false",
        help="Disable LoRA (enabled by default)",
    )
    p.set_defaults(use_lora=False)
    p.add_argument("--predict",                    type=str,   default=None)
    args = p.parse_args()

    # ─── 2) Data ──────────────────────────────────────────────
    ds, labels = build_dataset(sample_finer=args.sample_finer)
    label_list_names = labels
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

    # ─── 3) Model ─────────────────────────────────────────────
    if args.use_lora:
        # 4-bit quant + LoRA
        print("ATTEMPTING TO USE LORA")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForTokenClassification.from_pretrained(
        "nlpaueb/legal-bert-base-uncased",
        num_labels=len(labels),
        quantization_config=bnb_config,
        
        )   
        if not PEFT:
            raise RuntimeError("peft not installed; rerun without --no_lora or install it")

        model = get_peft_model(
            model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["query","value"],
                lora_dropout=0.05,
            ),
        )
        model.gradient_checkpointing_enable()

        # Debug: how many params require grad?
        total, trainable = 0, 0
        for n, p in model.named_parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        print(f">>> model params: {total:,}, trainable: {trainable:,}")
    else:
        print("NOT GOING TO USE LORA")

        model = AutoModelForTokenClassification.from_pretrained(
            "nlpaueb/legal-bert-base-uncased",
            num_labels=len(labels),
        )
        model.gradient_checkpointing_enable()

    # ─── 4) Tokenize ──────────────────────────────────────────
    tok_ds = ds.map(
        lambda ex: tokenize_and_align(ex, tokenizer, args.max_length),
        batched=True,
        batch_size=64,
        remove_columns=ds["train"].column_names,
    )
    collator = DataCollatorForTokenClassification(tokenizer)

    # ─── 5) Metrics ───────────────────────────────────────────
    def metric_fn(p):
        preds = p.predictions.argmax(-1)
        true_seqs, pred_seqs = [], []
        for pred, lab in zip(preds, p.label_ids):
            true_seqs.append([label_list_names[i] for i, l in zip(pred, lab) if l != -100])
            pred_seqs.append([label_list_names[i] for i, l in zip(pred, lab) if l != -100])
        return {
            "precision": precision_score(true_seqs, pred_seqs),
            "recall":    recall_score(true_seqs, pred_seqs),
            "f1":        f1_score(true_seqs, pred_seqs),
            "accuracy":  accuracy_score(true_seqs, pred_seqs),
        }

    # ─── 6) TrainingArguments ─────────────────────────────────
    targs = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=tok_ds["train"],
        eval_dataset=tok_ds["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=metric_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ─── 7) Train or Predict ─────────────────────────────────
    if args.predict is None:
        trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)
        plot_history(trainer, Path(args.output_dir) / "training_curve.png")
        print(trainer.evaluate(tok_ds["test"], metric_key_prefix="test"))
    else:
        nlp = pipeline(
            "token-classification",
            model=args.output_dir,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
        )
        for ent in nlp(args.predict):
            print(ent)

if __name__ == "__main__":
    main()
