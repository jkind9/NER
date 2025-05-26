import os
import torch
import datasets
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from sklearn.metrics import precision_recall_fscore_support

# ====== METRICS ======
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

# ====== TOKENIZE & ALIGN LABELS ======
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
                label_ids.append(label[word_idx] if label[word_idx] != -100 else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# ====== DEVICE SETUP ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ====== LOAD DATASET & SPLIT ======
dataset = datasets.load_dataset("lawinsider/uk_ner_contracts")
val_test = dataset["validation"].train_test_split(test_size=0.5, seed=42)
dataset["validation"] = val_test["train"]
dataset["test"] = val_test["test"]





#Base model COmparison
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification

# Load base model (random classifier head)
model_checkpoint = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
test_dataset = tokenized_datasets["test"]

base_model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=tokenized_datasets["train"].features["ner_tags"].feature.num_classes
).to(device)
base_trainer = Trainer(
    model=base_model,
    args=TrainingArguments(output_dir="./tmp_base_eval", per_device_eval_batch_size=16),
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics,
)

base_results = base_trainer.evaluate(tokenized_datasets["test"])
print("Base Legal-BERT (no fine-tuning) on test set:")
print(f"F1 = {base_results['eval_f1']:.4f}, Precision = {base_results['eval_precision']:.4f}, Recall = {base_results['eval_recall']:.4f}")


# # ====== LOAD TOKENIZER ======
# model_checkpoint = "nlpaueb/legal-bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
# test_dataset = tokenized_datasets["test"]
# # ====== EVALUATE EACH CHECKPOINT ======
# from transformers import TrainingArguments

# training_args = TrainingArguments(output_dir="./tmp_eval", per_device_eval_batch_size=16)
# data_collator = DataCollatorForTokenClassification(tokenizer)
# test_results = []

# checkpoints_dir = "./ner_model"
# for subdir in sorted(os.listdir(checkpoints_dir)):
#     if not subdir.startswith("checkpoint"):
#         continue
#     path = os.path.join(checkpoints_dir, subdir)
#     epoch = int(subdir.split("-")[-1])
#     print(f"\n📊 Evaluating {subdir} on test set...")

#     model = AutoModelForTokenClassification.from_pretrained(path).to(device)
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         tokenizer=tokenizer,
#         data_collator=data_collator,
#         compute_metrics=compute_metrics,
#     )

#     result = trainer.evaluate(test_dataset)
#     result["epoch"] = epoch
#     test_results.append(result)

# # ====== PLOT F1 OVER EPOCHS ======
# epochs = [r["epoch"] for r in test_results]
# f1s = [r["eval_f1"] for r in test_results]

# plt.plot(epochs, f1s, marker="o")
# plt.title("Test F1 Score by Epoch Checkpoint")
# plt.xlabel("Epoch")
# plt.ylabel("F1 Score")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("test_f1_by_epoch.png")

# # ====== PRINT RESULTS ======
# for r in sorted(test_results, key=lambda x: x["epoch"]):
#     print(f"Epoch {r['epoch']}: F1 = {r['eval_f1']:.4f}, Precision = {r['eval_precision']:.4f}, Recall = {r['eval_recall']:.4f}")

# #####################################
