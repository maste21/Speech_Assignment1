"""
Q1 - Manual MFCC / Cepstrum Engine
Dataset: hf-internal-testing/librispeech_asr_demo (clean parquet, no scripts)
"""

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

# ─────────────────────────────────────────
# 1. Load one sample
# ─────────────────────────────────────────
print("Loading dataset...")
ds     = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation[:1]")
sample = ds[0]
audio  = np.array(sample["audio"]["array"], dtype=np.float32)
sr     = sample["audio"]["sampling_rate"]
print(f"Sample rate: {sr} Hz  |  Duration: {len(audio)/sr:.2f}s")

# ─────────────────────────────────────────
# STEP 1 – Pre-emphasis
# ─────────────────────────────────────────
def pre_emphasis(signal, coeff=0.97):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

# ─────────────────────────────────────────
# STEP 2 – Framing + Windowing (Hamming)
# ─────────────────────────────────────────
def frame_signal(signal, sr, frame_ms=25, step_ms=10):
    frame_len  = int(sr * frame_ms / 1000)
    frame_step = int(sr * step_ms  / 1000)
    num_frames = 1 + (len(signal) - frame_len) // frame_step
    if num_frames <= 0:
        num_frames = 1
    frames = []
    for i in range(num_frames):
        start = i * frame_step
        end   = start + frame_len
        frame = signal[start:end]
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)))
        frames.append(frame * np.hamming(frame_len))
    return np.stack(frames)

# ─────────────────────────────────────────
# STEP 3 – FFT Power Spectrum
# ─────────────────────────────────────────
def power_spectrum(frames, NFFT=512):
    mag = np.abs(np.fft.rfft(frames, n=NFFT))
    return (1.0 / NFFT) * (mag ** 2)

# ─────────────────────────────────────────
# STEP 4 – Mel Filterbank
# ─────────────────────────────────────────
def hz_to_mel(hz):  return 2595 * np.log10(1 + hz / 700)
def mel_to_hz(mel): return 700  * (10 ** (mel / 2595) - 1)

def mel_filterbank(sr, NFFT=512, n_filters=26, fmin=0, fmax=None):
    if fmax is None:
        fmax = sr / 2
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_filters + 2)
    hz_pts  = mel_to_hz(mel_pts)
    bin_pts = np.floor((NFFT + 1) * hz_pts / sr).astype(int)
    fb = np.zeros((n_filters, NFFT // 2 + 1))
    for m in range(1, n_filters + 1):
        l, c, r = bin_pts[m-1], bin_pts[m], bin_pts[m+1]
        for k in range(l, c):
            fb[m-1, k] = (k - l) / (c - l + 1e-8)
        for k in range(c, r):
            fb[m-1, k] = (r - k) / (r - c + 1e-8)
    return fb

# ─────────────────────────────────────────
# STEP 5 – Log + DCT → MFCC
# ─────────────────────────────────────────
def dct_manual(x, n_ceps=13):
    N  = len(x)
    ks = np.arange(N)
    return np.array([
        np.sum(x * np.cos(np.pi * n * (2*ks + 1) / (2*N)))
        for n in range(n_ceps)
    ])

def compute_mfcc(signal, sr, n_mfcc=13, n_filters=26, NFFT=512):
    sig    = pre_emphasis(signal)
    frames = frame_signal(sig, sr)
    ps     = power_spectrum(frames, NFFT)
    fb     = mel_filterbank(sr, NFFT, n_filters)
    mel_e  = np.dot(ps, fb.T)
    log_e  = np.log(mel_e + 1e-8)
    mfcc   = np.array([dct_manual(row, n_mfcc) for row in log_e])
    return mfcc

# ─────────────────────────────────────────
# Run & Plot
# ─────────────────────────────────────────
print("Computing MFCC...")
mfcc = compute_mfcc(audio, sr)
print(f"MFCC shape: {mfcc.shape}")

plt.figure(figsize=(10, 4))
plt.imshow(mfcc.T, aspect="auto", origin="lower", cmap="viridis")
plt.colorbar(label="Amplitude")
plt.title("Manual MFCC (No Librosa)")
plt.xlabel("Frame")
plt.ylabel("Coefficient")
plt.tight_layout()
plt.savefig("mfcc_output.png", dpi=150)
plt.show()
print("Done! Saved → mfcc_output.png")
