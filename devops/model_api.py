from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from collections import Counter
import uvicorn
import pandas as pd

# === LOAD MODEL & TOKENIZER ===
from datasets import load_dataset

# Load model and tokenizer with mappings
tokenizer = AutoTokenizer.from_pretrained("Trad_NER/final_model")
model = AutoModelForTokenClassification.from_pretrained(
    "Trad_NER/final_model",
)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
# === FASTAPI SETUP ===
app = FastAPI()

class NERInput(BaseModel):
    text: str

@app.post("/ner")
async def extract_entities(data: NERInput):
    ner_results = ner_pipeline(data.text)

    # Extract entities and types
    entities = [(ent["word"], ent["entity_group"]) for ent in ner_results]
    counter = Counter(entities)

    # Build DataFrame-style result
    df = pd.DataFrame([
        {"entity": ent[0], "type": ent[1], "count": count}
        for ent, count in counter.items()
    ])

    # Return as list of dicts (JSON serializable)
    return df.to_dict(orient="records")

# === LOCAL TESTING ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
