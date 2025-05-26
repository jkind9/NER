#!/usr/bin/env python
"""train_uk_legal_ner.py

Fine‑tune a legal RoBERTa model on UK legal contract NER data (or run inference).

Usage:

# Training
python train_uk_legal_ner.py \
    --mode train \
    --dataset_name lawinsider/uk_ner_contracts_spacy \
    --model_name_or_path lexlms/legal-roberta-base \
    --output_dir ./uk_legal_ner_checkpoint

# Prediction (after training)
python train_uk_legal_ner.py \
    --mode predict \
    --output_dir ./uk_legal_ner_checkpoint \
    --text "This Master Services Agreement (MSA) shall ..."

Requirements:
  pip install -U "transformers>=4.39.0" datasets accelerate peft evaluate seqeval
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["train", "predict"], required=True, help="train or predict"
    )
    parser.add_argument(
        "--dataset_name", default="lawinsider/uk_ner_contracts_spacy", help="HF dataset id"
    )
    parser.add_argument(
        "--model_name_or_path",
        default="lexlms/legal-roberta-base",
        help="pre‑trained checkpoint",
    )
    parser.add_argument("--output_dir", default="uk_legal_ner_checkpoint")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--text", help="raw text for inference")
    parser.add_argument("--max_length", type=int, default=512)
    return parser.parse_args()


def collect_labels(ds):
    labels = set()
    for ents in ds["train"]["entities"]:
        labels.update(e["label"] for e in ents)
    return sorted(labels)


def build_label_maps(entity_labels):
    full = ["O"]
    for lab in entity_labels:
        full.extend([f"B-{lab}", f"I-{lab}"])
    label2id = {lab: i for i, lab in enumerate(full)}
    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label


def encode_batch(examples, tokenizer, label2id, max_length):
    encodings = tokenizer(
        examples["text"], truncation=True, max_length=max_length, return_offsets_mapping=True
    )
    batch_labels = []
    for offsets, entities in zip(encodings["offset_mapping"], examples["entities"]):
        # initialise with O
        seq_labels = [label2id["O"]] * len(offsets)
        for ent in entities:
            for idx, (start, end) in enumerate(offsets):
                if start >= ent["start"] and end <= ent["end"] and start != end:
                    prefix = "B" if start == ent["start"] else "I"
                    seq_labels[idx] = label2id[f"{prefix}-{ent['label']}"]

        # mask special tokens
        seq_labels = [
            (lab if offsets[i][0] != offsets[i][1] else -100) for i, lab in enumerate(seq_labels)
        ]
        batch_labels.append(seq_labels)

    encodings["labels"] = batch_labels
    return encodings


def main():
    args = parse_args()

    if args.mode == "train":
        ds = load_dataset(args.dataset_name)

        entity_labels = collect_labels(ds)
        label2id, id2label = build_label_maps(entity_labels)

        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        base_model = AutoModelForTokenClassification.from_pretrained(
            args.model_name_or_path, num_labels=len(label2id), id2label=id2label, label2id=label2id
        )

        # LoRA for parameter‑efficient fine‑tuning
        base_model = prepare_model_for_kbit_training(base_model)
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=["query", "value"],
            task_type="TOKEN_CLS",
        )
        model = get_peft_model(base_model, config)

        tokenized = ds.map(
            lambda ex: encode_batch(ex, tokenizer, label2id, args.max_length),
            batched=True,
            remove_columns=ds["train"].column_names,
        )

        collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)
        metric = load_metric("seqeval")

        def compute_metrics(p):
            preds, labels = p
            preds = np.argmax(preds, axis=2)
            true_preds, true_labels = [], []
            for pred_row, lab_row in zip(preds, labels):
                row_preds, row_labels = [], []
                for p_id, l_id in zip(pred_row, lab_row):
                    if l_id == -100:
                        continue
                    row_preds.append(id2label[p_id])
                    row_labels.append(id2label[l_id])
                true_preds.append(row_preds)
                true_labels.append(row_labels)
            res = metric.compute(predictions=true_preds, references=true_labels)
            return {
                "precision": res["overall_precision"],
                "recall": res["overall_recall"],
                "f1": res["overall_f1"],
                "accuracy": res["overall_accuracy"],
            }

        args_train = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            evaluation_strategy="epoch",
            logging_steps=50,
            save_total_limit=1,
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=True,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=args_train,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized.get("validation"),
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"✔ Model saved to {args.output_dir}")

    else:
        if not args.text:
            raise ValueError("--text must be supplied for prediction mode")
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        model = AutoModelForTokenClassification.from_pretrained(args.output_dir)
        model.eval()
        enc = tokenizer(args.text, return_offsets_mapping=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
        pred_ids = logits.softmax(-1).argmax(-1)[0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        offsets = enc["offset_mapping"][0].tolist()

        ent_spans = []
        current = None
        for tok, idx, (start, end) in zip(tokens, pred_ids, offsets):
            label = model.config.id2label[idx]
            if label.startswith("B-"):
                if current:
                    ent_spans.append(current)
                current = {"label": label[2:], "start": start, "end": end}
            elif label.startswith("I-") and current and label[2:] == current["label"]:
                current["end"] = end
            else:
                if current:
                    ent_spans.append(current)
                    current = None
        if current:
            ent_spans.append(current)

        print(json.dumps(ent_spans, indent=2))


if __name__ == "__main__":
    main()
