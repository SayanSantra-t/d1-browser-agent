import os
import sys
import time
import torch
from transformers import AutoModel, AutoTokenizer

model_path = r"C:\Users\sayan\models\d1-omni-600M"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    dtype=dtype
).to(device)

state = {
    "goal": "Search for Python machine learning tutorials",
    "url": "https://www.google.com"
}

questions = {
    "target_element": {
        "type": "choice",
        "instructions": "Which element should be clicked or focused to search for machine learning tutorials?",
        "criteria": {
            "e1": "button 'Google Search'",
            "e2": "textbox 'Search query input box'",
            "e3": "link 'Gmail'",
            "e4": "link 'Images'"
        }
    }
}

# Warmup run
with torch.no_grad():
    _ = model.system_one(state, questions)

# Benchmark 3 warm runs
latencies = []
for i in range(3):
    t0 = time.time()
    with torch.no_grad():
        res = model.system_one(state, questions)
    lat = (time.time() - t0) * 1000.0
    latencies.append(lat)
    print(f"Run {i+1}: {lat:.2f} ms")

print(f"\nAverage warm GPU decision latency: {sum(latencies)/len(latencies):.2f} ms!")
