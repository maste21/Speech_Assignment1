"""
Q1 - Spectral Leakage & SNR Analysis
Dataset: hf-internal-testing/librispeech_asr_demo
"""

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

print("Loading dataset...")
ds    = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation[:1]")
audio = np.array(ds[0]["audio"]["array"], dtype=np.float32)
sr    = ds[0]["audio"]["sampling_rate"]

frame_len = int(sr * 0.025)
segment   = audio[:frame_len]

windows = {
    "Rectangular": np.ones(frame_len),
    "Hamming":     np.hamming(frame_len),
    "Hanning":     np.hanning(frame_len),
}
NFFT = 512

def compute_snr(spectrum):
    peak        = np.max(spectrum)
    noise_floor = np.mean(spectrum)
    return 10 * np.log10(peak / (noise_floor + 1e-10))

def compute_leakage(spectrum):
    threshold = np.percentile(spectrum, 95)
    main_lobe = np.sum(spectrum[spectrum >= threshold])
    total     = np.sum(spectrum) + 1e-10
    return 1.0 - (main_lobe / total)

results = {}
spectra = {}
for name, win in windows.items():
    windowed      = segment * win
    mag           = np.abs(np.fft.rfft(windowed, n=NFFT)) ** 2
    spectra[name] = mag
    results[name] = {
        "SNR (dB)":      round(compute_snr(mag), 3),
        "Leakage Ratio": round(compute_leakage(mag), 5),
    }

print("\n" + "="*45)
print(f"{'Window':<15} {'SNR (dB)':>12} {'Leakage Ratio':>15}")
print("-" * 45)
for name, vals in results.items():
    print(f"{name:<15} {vals['SNR (dB)']:>12} {vals['Leakage Ratio']:>15}")
print("=" * 45)

freqs = np.fft.rfftfreq(NFFT, d=1/sr)
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
colors = ["steelblue", "darkorange", "green"]

for ax, (name, mag), color in zip(axes, spectra.items(), colors):
    ax.plot(freqs, 10 * np.log10(mag + 1e-10), color=color, linewidth=0.8)
    ax.set_title(f"{name}\nSNR={results[name]['SNR (dB)']} dB | Leakage={results[name]['Leakage Ratio']}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.grid(alpha=0.3)

plt.suptitle("Spectral Leakage & SNR — Three Windowing Functions", fontsize=13)
plt.tight_layout()
plt.savefig("leakage_snr_plot.png", dpi=150)
plt.show()
print("Done! Saved → leakage_snr_plot.png")
