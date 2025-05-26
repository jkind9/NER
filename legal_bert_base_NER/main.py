from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
def group_entities(tokens):
    entities = []
    current = None

    for tok in tokens:
        label = tok["entity"]
        word = tok["word"]
        if label == "O":
            if current:
                entities.append(current)
                current = None
            continue

        tag, ent_type = label.split("-") if "-" in label else ("O", None)

        if tag == "B":
            if current:
                entities.append(current)
            current = {"entity": ent_type, "word": word, "score": [tok["score"]], "start": tok["start"], "end": tok["end"]}
        elif tag == "I" and current and current["entity"] == ent_type:
            current["word"] += " " + word
            current["score"].append(tok["score"])
            current["end"] = tok["end"]
        else:
            if current:
                entities.append(current)
            current = None  # reset on unexpected I- tag

    if current:
        entities.append(current)

    # Average confidence scores
    for ent in entities:
        ent["score"] = sum(ent["score"]) / len(ent["score"])

    return entities


# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("easwar03/legal-bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("easwar03/legal-bert-base-NER")

# No aggregation
ner = pipeline("ner", model=model, tokenizer=tokenizer)

text = "The agreement between ACM and John Doe was signed in New York on January 5, 2021."
output = ner(text)

# Step 1: Map LABEL_x to B-XXX, I-XXX
id2label = {
    0: "B-LOC",
    1: "I-LOC",
    2: "B-ORG",
    3: "B-PER",
    4: "I-LOC",
    5: "I-PER",
    6: "I-ORG",
    7: "I-PER",
    8: "O"
}


for tok in output:
    label_id = int(tok["entity"].split("_")[-1])
    tok["entity"] = id2label[label_id]

# Step 2: Group tokens into spans
grouped = group_entities(output)

# Step 3: Print results
for ent in grouped:
    print(f"{ent['word']} ({ent['entity']}) - {ent['score']:.4f}")