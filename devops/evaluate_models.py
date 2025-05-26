import os
import torch
import datasets
import numpy as np
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import ast

def safe_literal_eval(val):
    if isinstance(val, str):
        return ast.literal_eval(val)
    return val

def map_labels_to_ids(label2id, df, label_col="ner_tags"):
    def convert(labels):
        return [label2id[label] for label in labels]
    df[label_col] = df[label_col].apply(convert)

# ========== METRICS ==========
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = precision_recall_fscore_support(
        sum(true_labels, []), sum(true_predictions, []), average="weighted"
    )
    return {
        "precision": results[0],
        "recall": results[1],
        "f1": results[2],
    }

# ========== TOKENIZATION ==========
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# ========== TEST SINGLE EXAMPLE ==========
def test_model(tokenized_test_dataset, model, device):
    first_example = tokenized_test_dataset[0]

    input_keys = ["input_ids", "attention_mask"]
    if "token_type_ids" in first_example:
        input_keys.append("token_type_ids")

    inputs = {k: torch.tensor([first_example[k]]).to(device) for k in input_keys}

    # Instead of calling tokenizer.word_ids(), call it on tokenized inputs:
    # We need to re-run the tokenizer on the original tokens with is_split_into_words=True
    # because tokenized_test_dataset no longer has word_ids info.

    original_tokens = first_example["tokens"]
    encoding = tokenizer(original_tokens, is_split_into_words=True, return_tensors="pt")
    word_ids = encoding.word_ids(batch_index=0)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    predicted_label_ids = outputs.logits.argmax(dim=-1).squeeze().tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())

    previous_word_idx = None
    aligned_predictions = []
    aligned_tokens = []

    for pred_id, word_idx, token in zip(predicted_label_ids, word_ids, tokens):
        if word_idx is None or word_idx == previous_word_idx:
            continue
        aligned_predictions.append(model.config.id2label[pred_id])
        aligned_tokens.append(token)
        previous_word_idx = word_idx

    print("\nToken : Predicted Label")
    for token, label in zip(aligned_tokens, aligned_predictions):
        print(f"{token}: {label}")


# ========== MAIN ==========
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

    model = AutoModelForTokenClassification.from_pretrained(
        r"C:\Users\jkind\Documents\NER\Trad_NER\final_model",
    ).to(device)

    id2label = model.config.id2label
    label2id = model.config.label2id

    test_df = pd.read_csv(r"C:\Users\jkind\Documents\NER\Trad_NER\TestData.csv")
    print("Columns in test_df:", test_df.columns.tolist())

    test_df = test_df.drop(columns=["Unnamed: 0", "sentence_id"], errors="ignore")
    test_df = test_df.rename(columns={"token": "tokens", "ner_tag": "ner_tags"})

    test_df["tokens"] = test_df["tokens"].apply(safe_literal_eval)
    test_df["ner_tags"] = test_df["ner_tags"].apply(safe_literal_eval)

    map_labels_to_ids(label2id, test_df, "ner_tags")

    test_dataset = datasets.Dataset.from_pandas(test_df)

    tokenized_test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)

    # test_model(tokenized_test_dataset=tokenized_test_dataset, model=model, device=device)

    # You can uncomment below to run full evaluation if needed:
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
        args=TrainingArguments(output_dir="./tmp_eval", per_device_eval_batch_size=16),
    )
    results = trainer.evaluate(tokenized_test_dataset)
    print(f"\nEvaluation results on TestData.csv:\nF1: {results['eval_f1']:.4f}\nPrecision: {results['eval_precision']:.4f}\nRecall: {results['eval_recall']:.4f}")
    from collections import Counter
    all_labels = sum(test_dataset["ner_tags"], [])
    label_counts = Counter(all_labels)
    print(label_counts)
