This project is focused on testing and exploring frameworks for name entity recognition in the legal domain.

Several possibilities exist, including usage of encoder frameworks such as BERT however these methods are now outdated.

Future development should include a unification of model testing, to compare different frameworks (both base and fine-tuned) in the same script.

SOTA method seems to be legal-bert trained on the legal-ner dataset (which produces the model "en_legal_ner_trf" in spacy) achieving F1 of 0.9.

Potential improvements in F1 could include combining datasets, potentially considering multilanguage models and including the german dataset: https://huggingface.co/datasets/elenanereiss/german-ler

Datasets of interest include:

https://github.com/Legal-NLP-EkStep/legal_NER
https://huggingface.co/datasets/joelniklaus/mapa
https://huggingface.co/datasets/nlpaueb/finer-139

Libraries/Projects of interest include:

https://github.com/ICLRandD/Blackstone
