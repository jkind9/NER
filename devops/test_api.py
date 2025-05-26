import requests
import pandas as pd

def test_ner_api(text: str, host: str = "http://localhost", port: int = 8000) -> pd.DataFrame:
    url = f"{host}:{port}/ner"
    payload = {"text": text}
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        return pd.DataFrame(result)
    
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        return pd.DataFrame()

df = test_ner_api("James Alexander signed a contract with Microsoft Inc. in London on 5th May 2023.")
print(df)