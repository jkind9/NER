# import nltk
# nltk.download("punkt_tab")
# nltk.download("maxent_ne_chunker_tab")
# nltk.download("words")
# nltk.download("averaged_perceptron_tagger_eng")
# import spacy
# import torch
# from transformers import AutoTokenizer, AutoModelForTokenClassification
# from transformers import pipeline
# from nltk import word_tokenize, pos_tag, ne_chunk
# from collections import Counter
# import re


# ----------- 1. spaCy -----------
import spacy
import pandas as pd
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

model_name = "nlpaueb/legal-bert-base-uncased"  # Replace with legal NER fine-tuned version if needed

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Read file content
text = Path("miranda_v_arizona.txt").read_text(encoding="utf-8")
# Run NER


# Run NER
entities = ner_pipeline(text)



# Count entities by (text, label)
ent_counter = Counter(entities)

# Convert to DataFrame
df = pd.DataFrame([
    {"Entity": ent_text, "Type": ent_label, "Count": count}
    for (ent_text, ent_label), count in ent_counter.items()
])

# Sort by count
df = df.sort_values(by="Count", ascending=False)

# Display
print(df.to_string(index=False))

# ----------- 2. NLTK -----------

# tokens = word_tokenize(text)
# tags = pos_tag(tokens)
# tree = ne_chunk(tags)
# nltk_ents = []
# for subtree in tree:
#     if hasattr(subtree, 'label'):
#         nltk_ents.append((" ".join(token for token, _ in subtree), subtree.label()))
# nltk_counter = Counter(ent[1] for ent in nltk_ents)

# # ----------- 3. HuggingFace LegalBERT NER -----------
# tokenizer = AutoTokenizer.from_pretrained("easwar03/legal-bert-base-NER")
# model = AutoModelForTokenClassification.from_pretrained("easwar03/legal-bert-base-NER")
# hf_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# hf_results = hf_pipeline(text)
# hf_ents = [(ent['word'], ent['entity_group']) for ent in hf_results]
# hf_counter = Counter(ent[1] for ent in hf_ents)

# ----------- Print Comparison -----------
# print("\nspaCy Entities (top 5):", spacy_ents[:5])
# print("spaCy Entity Count by Type:", spacy_counter)

# print("\nNLTK Entities (top 5):", nltk_ents[:5])
# print("NLTK Entity Count by Type:", nltk_counter)

