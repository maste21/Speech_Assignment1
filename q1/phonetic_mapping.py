"""
Q1 - Phonetic Mapping via Wav2Vec2 + RMSE
Dataset: hf-internal-testing/librispeech_asr_demo
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

print("Loading dataset...")
ds    = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation[:3]")
audio = np.array(ds[2]["audio"]["array"], dtype=np.float32)
sr    = ds[2]["audio"]["sampling_rate"]

boundaries = np.load("boundaries.npy")
frame_step = int(np.load("frame_step.npy"))
manual_boundaries_samples = boundaries * frame_step

print("Loading Wav2Vec2...")
MODEL_ID  = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model     = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
model.eval()

inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
with torch.no_grad():
    logits = model(**inputs).logits

predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.decode(predicted_ids[0])
print(f"Transcription: {transcription}")

model_stride = 320
tokens       = predicted_ids[0].numpy()
token_boundaries = [0]
for i in range(1, len(tokens)):
    if tokens[i] != tokens[i-1]:
        token_boundaries.append(i)
token_boundaries.append(len(tokens))
model_boundaries_samples = np.array(token_boundaries) * model_stride

def match_boundaries(manual, model_b):
    rmse_list = []
    for m in manual:
        nearest = model_b[np.argmin(np.abs(model_b - m))]
        rmse_list.append((m - nearest) ** 2)
    return np.sqrt(np.mean(rmse_list))

manual_inner = manual_boundaries_samples[1:-1]
model_inner  = model_boundaries_samples[1:-1]

if len(manual_inner) > 0 and len(model_inner) > 0:
    rmse_samples = match_boundaries(manual_inner, model_inner)
    rmse_ms      = (rmse_samples / sr) * 1000
else:
    rmse_samples, rmse_ms = 0, 0

print(f"\n{'='*40}")
print(f"Manual boundaries : {len(manual_inner)}")
print(f"Model  boundaries : {len(model_inner)}")
print(f"RMSE (samples)    : {rmse_samples:.2f}")
print(f"RMSE (ms)         : {rmse_ms:.2f} ms")
print("=" * 40)

t = np.arange(len(audio)) / sr
plt.figure(figsize=(12, 4))
plt.plot(t, audio, color="steelblue", linewidth=0.5, label="Waveform")
for i, s in enumerate(manual_boundaries_samples):
    plt.axvline(s/sr, color="green", linewidth=0.8, alpha=0.7, label="Manual" if i==0 else "")
for i, s in enumerate(model_boundaries_samples):
    plt.axvline(s/sr, color="red", linestyle="--", linewidth=0.8, alpha=0.6, label="Wav2Vec2" if i==0 else "")
plt.title(f"Boundary Comparison  |  RMSE = {rmse_ms:.2f} ms\nTranscription: '{transcription}'")
plt.xlabel("Time (s)")
plt.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig("phonetic_mapping.png", dpi=150)
plt.show()
print("Done! Saved → phonetic_mapping.png")
