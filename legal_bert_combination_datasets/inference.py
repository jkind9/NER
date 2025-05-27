from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import nltk

# Load tokenizer and model (from your trained checkpoint)
model_path = r"C:\Users\jkind\Documents\NER\eng_legal_ner_union"  # your fine-tuned model directory
model_checkpoint = r"nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_path)
model.eval()
print(model.config.id2label)

import nltk
nltk.download("punkt")

example = "James Alexander was going to do business with OpenAI in New York."

words = nltk.word_tokenize(example)

inputs = tokenizer(words, is_split_into_words=True, return_tensors="pt")
outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()
word_ids = inputs.word_ids(batch_index=0)

predicted_labels = []
previous_word_idx = None
for pred, word_idx in zip(predictions, word_ids):
    if word_idx is None or word_idx == previous_word_idx:
        continue
    predicted_labels.append(model.config.id2label[pred])

print(list(zip(words, predicted_labels)))



