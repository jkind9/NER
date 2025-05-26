import json

meta = {
    "lang": "en",
    "name": "legal_ner_trf",
    "version": "3.2.0",
    "description": "Indian Legal Named Entity Recognition: Identifying relevant entities in an Indian legal document",
    "author": "Aman Tiwari",
    "email": "aman.tiwari@thoughtworks.com",
    "url": "https://github.com/Legal-NLP-EkStep/legal_NER",
    "license": "MIT",
    "spacy_version": ">=3.2.2,<3.3.0",
    "vectors": {"width": 0, "vectors": 0, "keys": 0, "name": None},
    "labels": {
        "transformer": [],
        "ner": [
            "CASE_NUMBER", "COURT", "DATE", "GPE", "JUDGE", "LAWYER", "ORG",
            "OTHER_PERSON", "PETITIONER", "PRECEDENT", "PROVISION", "RESPONDENT",
            "STATUTE", "WITNESS"
        ]
    },
    "pipeline": ["transformer", "ner"],
    "components": ["transformer", "ner"],
    "disabled": [],
    "requirements": ["spacy-transformers>=1.1.4,<1.2.0"]
}

with open("./en_legal_ner_trf/meta.json", "w") as f:
    json.dump(meta, f, indent=2)