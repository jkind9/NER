import datasets

finer_train = datasets.load_dataset("nlpaueb/finer-139", split="train")

finer_tag_names = finer_train.features["ner_tags"].feature.names
print(finer_tag_names)

data = {int(i):finer_tag_names[i] for i in range(len(finer_tag_names)) }

import json
with open("finer_mapping.json", "w+") as f:
    f.write(json.dumps(data))