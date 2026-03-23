"""
Q3 - Privacy Preserving Voice Transformation
Dataset: hf-internal-testing/librispeech_asr_demo
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datasets import load_dataset

def pitch_shift(audio, sr, semitones):
    factor   = 2 ** (semitones / 12.0)
    new_len  = int(len(audio) / factor)
    indices  = np.linspace(0, len(audio)-1, new_len)
    shifted  = np.interp(indices, np.arange(len(audio)), audio)
    indices2 = np.linspace(0, len(shifted)-1, len(audio))
    return np.interp(indices2, np.arange(len(shifted)), shifted).astype(np.float32)

def formant_scale(audio, factor=1.2, NFFT=512):
    hop     = 160
    results = []
    pad     = np.zeros(NFFT // 2)
    audio_p = np.concatenate([pad, audio, pad])
    for i in range(0, len(audio_p) - NFFT, hop):
        frame = audio_p[i:i+NFFT] * np.hamming(NFFT)
        spec  = np.fft.rfft(frame)
        mag   = np.abs(spec)
        phase = np.angle(spec)
        freqs = np.arange(len(mag))
        new_mag = np.interp(freqs, freqs/factor, mag, left=0, right=0)
        results.append(np.fft.irfft(new_mag * np.exp(1j * phase))[:hop])
    out = np.concatenate(results)
    return out[:len(audio)].astype(np.float32)

PROFILES = {
    "male_to_female": {"pitch_st": +3,  "formant": 1.15},
    "female_to_male": {"pitch_st": -3,  "formant": 0.85},
    "old_to_young":   {"pitch_st": +2,  "formant": 1.10},
    "young_to_old":   {"pitch_st": -2,  "formant": 0.90},
}

def transform_voice(audio, sr, profile_name):
    p      = PROFILES[profile_name]
    audio  = pitch_shift(audio, sr, p["pitch_st"])
    audio  = formant_scale(audio, p["formant"])
    audio /= (np.max(np.abs(audio)) + 1e-8)
    return audio

class PrivacyModule(nn.Module):
    def __init__(self, profile_name="male_to_female"):
        super().__init__()
        self.profile = profile_name
        self.gain    = nn.Parameter(torch.tensor(1.0))
    def forward(self, audio_tensor):
        outputs = []
        for a in audio_tensor:
            t = transform_voice(a.detach().numpy(), 16000, self.profile)
            outputs.append(torch.tensor(t))
        return torch.stack(outputs) * self.gain

print("Loading sample...")
ds    = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation[:1]")
audio = np.array(ds[0]["audio"]["array"], dtype=np.float32)
sr    = ds[0]["audio"]["sampling_rate"]

print("Applying all 4 voice transformations...")
fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
t = np.arange(len(audio)) / sr

axes[0].plot(t, audio, color="steelblue", linewidth=0.5)
axes[0].set_title("Original Audio")
axes[0].set_ylabel("Amplitude")

colors = ["#e74c3c", "#2ecc71", "#9b59b6", "#f39c12"]
for ax, (name, _), color in zip(axes[1:], PROFILES.items(), colors):
    transformed = transform_voice(audio, sr, name)
    noise = audio - transformed[:len(audio)]
    snr   = 10 * np.log10(np.mean(audio**2) / (np.mean(noise**2) + 1e-10))
    ax.plot(t, transformed, color=color, linewidth=0.5)
    ax.set_title(f"{name.replace('_',' ').title()}  |  SNR={snr:.1f} dB")
    ax.set_ylabel("Amplitude")

axes[-1].set_xlabel("Time (s)")
plt.suptitle("Privacy-Preserving Voice Transformations", fontsize=13)
plt.tight_layout()
plt.savefig("privacy_transform.png", dpi=150)
plt.show()
print("Done! Saved → privacy_transform.png")
