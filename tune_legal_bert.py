import datasets
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification, EarlyStoppingCallback
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torch
import os
import matplotlib.pyplot as plt

# ========== DEVICE ==========
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("❌ GPU not available, using CPU")

def print_device_stats():
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print(f"Allocated memory: {round(torch.cuda.memory_allocated() / 1024 ** 2, 2)} MB")
        print(f"Reserved memory: {round(torch.cuda.memory_reserved() / 1024 ** 2, 2)} MB")

print_device_stats()

# ========== LOAD DATA ==========
dataset = datasets.load_dataset("lawinsider/uk_ner_contracts")

# Split validation into validation/test
val_test = dataset["validation"].train_test_split(test_size=0.5, seed=42)
dataset["validation"] = val_test["train"]
dataset["test"] = val_test["test"]

# ========== LOAD MODEL ==========
model_checkpoint = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=dataset["train"].features["ner_tags"].feature.num_classes
)

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
                label_ids.append(label[word_idx] if label[word_idx] != -100 else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

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
    results = precision_recall_fscore_support(sum(true_labels, []), sum(true_predictions, []), average="weighted")
    return {
        "precision": results[0],
        "recall": results[1],
        "f1": results[2],
    }

# ========== TRAINING ARGS ==========
training_args = TrainingArguments(
    output_dir="./ner_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=5,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
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
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()

# ========== EVALUATE EACH CHECKPOINT ON TEST SET ==========
test_results = []
checkpoint_dir = training_args.output_dir

for subdir in sorted(os.listdir(checkpoint_dir)):
    path = os.path.join(checkpoint_dir, subdir)
    if not subdir.startswith("checkpoint"):
        continue
    epoch = int(subdir.split("-")[-1])  # Assumes format: checkpoint-{step}
    print(f"\n📊 Evaluating {subdir} on test set...")
    model = AutoModelForTokenClassification.from_pretrained(path)
    trainer.model = model
    result = trainer.evaluate(tokenized_datasets["test"])
    result["epoch"] = epoch
    test_results.append(result)

# ========== PLOT F1 SCORE BY EPOCH ==========
epochs = [r["epoch"] for r in test_results]
f1s = [r["eval_f1"] for r in test_results]

plt.plot(epochs, f1s, marker="o")
plt.title("Test F1 Score by Epoch Checkpoint")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.grid(True)
plt.tight_layout()
plt.savefig("test_f1_by_epoch.png")
plt.show()
