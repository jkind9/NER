import os
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    BertForTokenClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn
from transformers.modeling_outputs import TokenClassifierOutput
import random

WEIGHTED = False

# ========== CUSTOM MODEL WITH WEIGHTED LOSS ==========
class WeightedTokenClassificationModel(BertForTokenClassification):
    def __init__(self, config, class_weights=None):
        super().__init__(config)
        self.class_weights = class_weights
        self.loss_fct = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=-100)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# ========== TOKENIZATION FUNCTION ==========
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["token"], truncation=True, is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(examples["ner_tag"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

  
def downsample_O_class(df_grouped, label_to_reduce="O", reduction_fraction=0.1, seed=42):
    random.seed(seed)
    filtered_rows = []

    for _, row in df_grouped.iterrows():
        tokens = row["token"]
        labels = row["ner_tag"]

        # Find indices of "O" tokens and non-"O" tokens
        O_indices = [i for i, label in enumerate(labels) if label == label_to_reduce]
        non_O_indices = [i for i, label in enumerate(labels) if label != label_to_reduce]

        # Downsample O indices
        sampled_O_indices = random.sample(O_indices, int(len(O_indices) * reduction_fraction)) if O_indices else []

        # Combine sampled O tokens and all non-O tokens
        keep_indices = sorted(sampled_O_indices + non_O_indices)

        # Filter tokens and labels
        filtered_tokens = [tokens[i] for i in keep_indices]
        filtered_labels = [labels[i] for i in keep_indices]

        filtered_rows.append({
            "sentence_id": row["sentence_id"],
            "token": filtered_tokens,
            "ner_tag": filtered_labels,
        })

    return pd.DataFrame(filtered_rows)


def parse_ener_csv(file_path):
    tokens = []
    tags = []

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            token = row[0].strip()
            tag = row[1].strip()
            if token == "" or tag == "":
                continue
            tokens.append(token)
            tags.append(tag)

    df = pd.DataFrame({"token": tokens, "ner_tag": tags})

    sentence_id = -1
    sentence_ids = []
    for token in df["token"]:
        if token.upper() == "-DOCSTART-":
            sentence_id += 1
        sentence_ids.append(sentence_id)

    df["sentence_id"] = sentence_ids

    df = df[df["sentence_id"] >= 0]
    df = df[df["token"].str.upper() != "-DOCSTART-"]

    return df

# ========== REPRODUCIBILITY ==========
from random import seed as python_seed
seed = 42
python_seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ========== DEVICE ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")

# ========== LOAD DATA ==========
file_path = r"C:\Users\jkind\Documents\NER\Trad_NER\all.csv"
df = parse_ener_csv(file_path=file_path)

grouped = df.groupby("sentence_id").agg({
    "token": list,
    "ner_tag": list
}).reset_index()

train_df, temp_df = train_test_split(grouped, test_size=0.4, random_state=seed)
downsampled_train_df = downsample_O_class(train_df, reduction_fraction=0.3)  # Keep 30% of O tokens

val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed)

test_df.to_csv(r"C:\Users\jkind\Documents\NER\Trad_NER\TestData.csv", index=False)

dataset = DatasetDict({
    "train": Dataset.from_pandas(downsampled_train_df.drop(columns=["sentence_id"])),
    "validation": Dataset.from_pandas(val_df.drop(columns=["sentence_id"])),
    "test": Dataset.from_pandas(test_df.drop(columns=["sentence_id"]))
})

label_list = sorted(set(tag for tags in dataset["train"]["ner_tag"] for tag in tags))
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}

# Compute class weights for weighted loss
all_train_labels = sum(dataset["train"]["ner_tag"], [])
unique_labels = np.unique(all_train_labels)
print("Unique labels in training data:", unique_labels)

classes = np.array(unique_labels)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=all_train_labels)
class_weights = torch.tensor(weights, dtype=torch.float).to(device)

print("Class weights:", {id2label[i]: weights[i] for i in range(len(weights))})

# ========== TOKENIZER ==========
model_checkpoint = r"nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# ========== METRICS ==========
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [p for (p, l) in zip(pred, lbl) if l != -100]
        for pred, lbl in zip(predictions, labels)
    ]
    true_labels = [
        [l for (p, l) in zip(pred, lbl) if l != -100]
        for pred, lbl in zip(predictions, labels)
    ]
    results = precision_recall_fscore_support(
        sum(true_labels, []), sum(true_predictions, []), average="weighted"
    )
    return {
        "precision": results[0],
        "recall": results[1],
        "f1": results[2],
    }

# ========== MODEL INITIALIZATION ==========
if WEIGHTED:
    model = WeightedTokenClassificationModel.from_pretrained(
        model_checkpoint,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        class_weights=class_weights,
    ).to(device)
else:
    model =  AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    ).to(device)

# ========== TRAINING ARGS ==========
training_args = TrainingArguments(
    output_dir="./ner_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    logging_dir="./logs",
    report_to="none",
    save_total_limit=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    learning_rate=5e-6,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    weight_decay=0.01,
)

# ========== TRAINING ==========
data_collator = DataCollatorForTokenClassification(tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

trainer.train()

# ========== SAVE FINAL MODEL ==========
final_dir = r"C:\Users\jkind\Documents\NER\Trad_NER\final_model"
model.config.id2label = id2label
model.config.label2id = label2id
trainer.save_model(final_dir)
tokenizer.save_pretrained(final_dir)

print(f"\nTraining complete. Model saved to: {final_dir}")

# ========== TEST SET EVALUATION ==========
test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
print("\nTest Set Evaluation Metrics:")
for k, v in test_results.items():
    print(f"{k}: {v:.4f}")

# ========== PLOT LOSS CURVE ==========
log_history = trainer.state.log_history
train_loss = [entry["loss"] for entry in log_history if "loss" in entry and "epoch" in entry]
eval_loss = [entry["eval_loss"] for entry in log_history if "eval_loss" in entry]
epochs = [entry["epoch"] for entry in log_history if "loss" in entry and "epoch" in entry]
eval_epochs = [entry["epoch"] for entry in log_history if "eval_loss" in entry]

plt.figure()
plt.plot(epochs, train_loss, marker="o", label="Train Loss")
plt.plot(eval_epochs, eval_loss, marker="x", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(r"C:\Users\jkind\Documents\NER\Trad_NER\loss_curve.png")
