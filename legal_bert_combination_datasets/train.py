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

from pprint import pprint
try:
    from peft import LoraConfig, get_peft_model
    PEFT = True
except ImportError:
    PEFT = False
from collections import Counter
import json, os, logging
import torch.nn as nn

MAP_FILE = r"C:\Users\jkind\Documents\NER\legal_bert_combination_datasets\label_maps.json"
logger = logging.getLogger(__name__)

class WeightedNERTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        # keep a loss function ready on the correct device
        self.loss_fct = nn.CrossEntropyLoss(
            weight=class_weights.to(self.args.device),
            ignore_index=-100,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits               # (B, T, C)
        loss = self.loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1),
        )
        return (loss, outputs) if return_outputs else loss
    
def make_class_weights(dataset, num_labels, ignore_idx=-100):
    counts = Counter()
    for ex in dataset:
        counts.update([t for t in ex["labels"] if t != ignore_idx])
    total = sum(counts.values())
    # inverse frequency
    inv_freq = torch.tensor(
        [(total / counts.get(i, 1)) for i in range(num_labels)],
        dtype=torch.float32,
    )
    # normalise so mean == 1
    return inv_freq / inv_freq.mean()


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


# ----------------------------------------------------------------------
# Hard-coded map loader
# ----------------------------------------------------------------------
def load_manual_map() -> dict[str, dict] | None:
    if os.path.exists(MAP_FILE):
        with open(MAP_FILE, "r") as fh:
            data = json.load(fh)
            return data
    return None
MANUAL_MAPS = load_manual_map()
# ----------------------------------------------------------------------
# 1. InLegal loader  → tokens, ner_tags (str)
# ----------------------------------------------------------------------
def load_inlegal(base_dir: str) -> DatasetDict:
    ds = load_dataset(
        "arrow",
        data_files={
            "train": f"{base_dir}/train.arrow/*.arrow",
            "validation": f"{base_dir}/dev.arrow/*.arrow",
            "test": f"{base_dir}/test.arrow/*.arrow",
        },
    )
    id2name = {int(k): v for k, v in MANUAL_MAPS["inlegal"].items()}

    def convert(ex):
        ex["ner_tags"] = [id2name[i] for i in ex["ner_tags"]]
        return ex

    ds = ds.map(convert)
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in {"tokens","ner_tags"}])
    return ds

# ----------------------------------------------------------------------
# 2. MAPA loader  → tokens, ner_tags (str)
# ----------------------------------------------------------------------
def load_mapa() -> DatasetDict:
    ds = load_dataset("joelniklaus/mapa").filter(lambda e: e.get("language","en")=="en")
    # keep "tokens" + "coarse_grained"
    def rename(ex):
        ex["ner_tags"] = ex["coarse_grained"]
        return ex
    ds = ds.map(rename)
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in {"tokens","ner_tags"}])
    return ds

# ----------------------------------------------------------------------
# 3. Merge helper
# ----------------------------------------------------------------------
def cast_to_labels(ds: DatasetDict, all_labels: list[str]) -> DatasetDict:
    feat = Features({
        "tokens": ds["train"].features["tokens"],
        "ner_tags": Sequence(ClassLabel(names=all_labels)),
    })
    name2id = {n:i for i,n in enumerate(all_labels)}

    def to_ids(ex):
        ex["ner_tags"] = [name2id[t] for t in ex["ner_tags"]]
        return ex

    return ds.map(to_ids).cast(feat)

# ----------------------------------------------------------------------
# 4. Public function
# ----------------------------------------------------------------------
def build_dataset(seed: int = 42) -> tuple[DatasetDict, list[str]]:
    in_ds   = load_inlegal(base_dir=r"C:\Users\jkind\Documents\NER\local_data\inlegalner_tokenized")
    mapa_ds = load_mapa()

    # log raw vocabularies
    for name, ds in [("INLEGAL", in_ds), ("MAPA", mapa_ds)]:
        vocab = {t for split in ds for ex in ds[split]["ner_tags"] for t in ex}
        logger.info("[%s] tag vocab (%d): %s", name, len(vocab), sorted(vocab))

    # union of label strings
    label_set = sorted(
        {t for split in in_ds  for ex in in_ds[split]["ner_tags"]  for t in ex} |
        {t for split in mapa_ds for ex in mapa_ds[split]["ner_tags"] for t in ex}
    )
    logger.info("Unified label set (%d): %s", len(label_set), label_set)

    # cast each corpus to int-encoded tags sharing the same ClassLabel
    in_ds   = cast_to_labels(in_ds,   label_set)
    mapa_ds = cast_to_labels(mapa_ds, label_set)

    # concatenate train splits (keep individual dev/test for now)
    train_pool = concatenate_datasets([in_ds["train"], mapa_ds["train"]]).shuffle(seed=seed)
    merged = DatasetDict(
        train=train_pool,
        validation=concatenate_datasets([in_ds["validation"], mapa_ds["validation"]]),
        test=concatenate_datasets([in_ds["test"], mapa_ds["test"]]),
    )
    return merged, label_set

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
def complete_args():
    # ─── 1) Arguments ─────────────────────────────────────────

    p = argparse.ArgumentParser()
    p.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    p.add_argument(
        "--output_dir",
        default="./eng_legal_ner_union",
        help="Where to save checkpoints & final model",
    )
    # step‐based training (instead of epochs)
    p.add_argument(
        "--max_steps",
        type=int,
        default=40000,
        help="Total optimization steps (overrides num_train_epochs)",
    )
    p.add_argument(
        "--eval_steps",
        type=int,
        default=1000,
        help="Run evaluation every X steps",
    )
    p.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every X steps",
    )
    p.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=3,
        help="Accumulate gradients N times before stepping",
    )
    p.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate",
    )
    p.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size per GPU/CPU for training",
    )
    p.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Batch size per GPU/CPU for evaluation",
    )
    p.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Max token length for inputs",
    )
    p.add_argument(
        "--sample_finer",
        type=int,
        default=0,
        help="If >0, subsample FiNER to this many examples",
    )
    # LoRA enabled by default; pass --no_lora to disable
    p.add_argument(
        "--no_lora",
        dest="use_lora",
        action="store_false",
        help="Enable LoRA (disabled by default)",
    )
    p.set_defaults(use_lora=False)
    p.add_argument(
        "--predict",
        type=str,
        default=None,
        help="Run a prediction and exit, instead of training",
    )
    args = p.parse_args()
    return args

def main():
    args = complete_args()
    # ─── 1) Logger ──────────────────────────────────────────────

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=getattr(logging, args.log_level),
    )
    logger.info("Starting with log level %s", args.log_level)


    # ─── 2) Data ──────────────────────────────────────────────
    ds, labels = build_dataset()
    label_list_names = labels

    
    def count_labels(dataset):
        counter = Counter()
        for row in dataset["ner_tags"]:
            counter.update(row)
        return counter

    print("▶ Class distribution (train):")
    train_counts = count_labels(ds["train"])
    pprint({labels[k]: v for k, v in train_counts.items()})

    print("\n▶ Class distribution (validation):")
    val_counts = count_labels(ds["validation"])
    pprint({labels[k]: v for k, v in val_counts.items()})

    print("\n▶ Class distribution (test):")
    test_counts = count_labels(ds["test"])
    pprint({labels[k]: v for k, v in test_counts.items()})

    num_labels = len(labels)


    # ─── 3) Model ─────────────────────────────────────────────
    if args.use_lora:

        print("ATTEMPTING TO USE LORA")
        # 1) Prepare 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        # 2) Load the quantized model
        model = AutoModelForTokenClassification.from_pretrained(
            "nlpaueb/legal-bert-base-uncased",
            num_labels=len(labels),
            quantization_config=bnb_config,
        )
        # 3) Move to GPU if available
        if torch.cuda.is_available():
            model.to("cuda")

        if not PEFT:
            raise RuntimeError("peft not installed; rerun without --no_lora or install it")
        # 4) Apply LoRA adapters
        model = get_peft_model(
            model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["query", "value"],
                lora_dropout=0.05,
            ),
        )

        # Debug: count total vs. trainable parameters
        total, trainable = 0, 0
        for _, p in model.named_parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        print(f">>> model params: {total:,}, trainable: {trainable:,}")

    else:
        print("NOT USING LORA")

        model = AutoModelForTokenClassification.from_pretrained(
            "nlpaueb/legal-bert-base-uncased",
            num_labels=len(labels),
        )
        model.gradient_checkpointing_enable()

    # ─── 4) Tokenize ──────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

    tok_ds = ds.map(
        lambda ex: tokenize_and_align(ex, tokenizer, args.max_length),
        batched=True,
        batch_size=64,
        remove_columns=ds["train"].column_names,
    )
    collator = DataCollatorForTokenClassification(tokenizer)
    class_weights = make_class_weights(tok_ds["train"], num_labels)
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
        evaluation_strategy="steps",       
        eval_steps=args.eval_steps,         
        save_strategy="steps",             
        save_steps=args.save_steps,         
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        max_steps=args.max_steps,           # use max_steps instead of num_train_epochs
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = WeightedNERTrainer(
        model=model,
        args=targs,
        train_dataset=tok_ds["train"],
        eval_dataset=tok_ds["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=metric_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        class_weights=class_weights,      
    )

    # ─── 7) Train or Predict ─────────────────────────────────
    if args.predict is None:
        trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)
        plot_history(trainer, Path(args.output_dir) / "training_curve.png")
        print(trainer.evaluate(tok_ds["test"], metric_key_prefix="test"))
        with open(Path(args.output_dir) / "label2id.json", "w") as f:
            json.dump({l: i for i, l in enumerate(labels)}, f, indent=2)

        with open(Path(args.output_dir) / "id2label.json", "w") as f:
            json.dump({i: l for i, l in enumerate(labels)}, f, indent=2)
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
