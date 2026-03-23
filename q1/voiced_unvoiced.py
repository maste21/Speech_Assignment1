"""
Q1 - Voiced/Unvoiced Boundary Detection via Cepstrum
Dataset: hf-internal-testing/librispeech_asr_demo
"""

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

print("Loading dataset...")
ds    = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation[:3]")
audio = np.array(ds[2]["audio"]["array"], dtype=np.float32)
sr    = ds[2]["audio"]["sampling_rate"]

def frame_signal(signal, sr, frame_ms=25, step_ms=10):
    frame_len  = int(sr * frame_ms / 1000)
    frame_step = int(sr * step_ms  / 1000)
    num_frames = 1 + (len(signal) - frame_len) // frame_step
    frames = []
    for i in range(num_frames):
        start = i * frame_step
        frame = signal[start:start+frame_len]
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)))
        frames.append(frame * np.hamming(frame_len))
    return np.stack(frames), frame_len, frame_step

frames, frame_len, frame_step = frame_signal(audio, sr)
NFFT = 512

def cepstrum(frame):
    spectrum = np.fft.rfft(frame, n=NFFT)
    log_spec = np.log(np.abs(spectrum) + 1e-10)
    return np.fft.irfft(log_spec)

low_q_end    = int(0.002 * sr)
high_q_start = int(0.002 * sr)
high_q_end   = int(0.0167 * sr)

low_energy, high_energy, zcr_list = [], [], []
for frame in frames:
    ceps = cepstrum(frame)
    low_energy .append(np.sum(ceps[:low_q_end] ** 2))
    high_energy.append(np.sum(ceps[high_q_start:high_q_end] ** 2))
    zcr_list   .append(np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame)))

low_energy  = np.array(low_energy)
high_energy = np.array(high_energy)
zcr_arr     = np.array(zcr_list)

high_norm    = (high_energy - high_energy.min()) / (high_energy.ptp() + 1e-10)
zcr_norm     = (zcr_arr    - zcr_arr.min())     / (zcr_arr.ptp()     + 1e-10)
voiced_score = high_norm - 0.5 * zcr_norm
threshold    = np.mean(voiced_score)
is_voiced    = voiced_score > threshold

boundaries = [0]
for i in range(1, len(is_voiced)):
    if is_voiced[i] != is_voiced[i-1]:
        boundaries.append(i)
boundaries.append(len(is_voiced))

frame_times = np.array([i * frame_step / sr for i in range(len(is_voiced))])
print(f"Voiced frames  : {is_voiced.sum()}")
print(f"Unvoiced frames: {(~is_voiced).sum()}")
print(f"Transitions    : {len(boundaries)-2}")

fig, axes = plt.subplots(3, 1, figsize=(12, 8))
t = np.arange(len(audio)) / sr
axes[0].plot(t, audio, color="steelblue", linewidth=0.5)
axes[0].set_title("Waveform")
axes[0].set_ylabel("Amplitude")

axes[1].plot(frame_times, voiced_score, color="purple", linewidth=0.8, label="Voiced Score")
axes[1].axhline(threshold, color="red", linestyle="--", label=f"Threshold={threshold:.3f}")
axes[1].set_title("Voiced Score (High-Quefrency Energy − ZCR)")
axes[1].legend(fontsize=8)

colors_v = ["#2ecc71" if v else "#e74c3c" for v in is_voiced]
for i in range(len(is_voiced)):
    axes[2].axvspan(frame_times[i], frame_times[i] + frame_step/sr, color=colors_v[i], alpha=0.6)
axes[2].set_title("Voiced (green) / Unvoiced (red)")
axes[2].set_xlabel("Time (s)")
axes[2].set_yticks([])

plt.tight_layout()
plt.savefig("voiced_unvoiced.png", dpi=150)
plt.show()
print("Done! Saved → voiced_unvoiced.png")

np.save("boundaries.npy",  np.array(boundaries))
np.save("is_voiced.npy",   is_voiced)
np.save("frame_step.npy",  np.array(frame_step))
print("Saved boundaries.npy")
