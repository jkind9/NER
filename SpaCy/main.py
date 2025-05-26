"""
Invoke-WebRequest -Uri "https://huggingface.co/opennyaiorg/en_legal_ner_trf/resolve/main/en_legal_ner_trf-any-py3-none-any.whl" -OutFile "en_legal_ner_trf-any-py3-none-any.whl"
pip install .\en_legal_ner_trf-any-py3-none-any.whl
"""

import spacy
nlp = spacy.load("en_legal_ner_trf")
doc = nlp("The District Court of Chester decided James had had his code stolen from OpenAI.")
for ent in doc.ents:
    print("TEXT:", ent.text, "|LABEL:", ent.label_)