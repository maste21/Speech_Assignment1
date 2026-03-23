"""
Q2 - Disentangled Speaker Recognition
Dataset: hf-internal-testing/librispeech_asr_demo (clean parquet)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

BATCH_SIZE = 16
EPOCHS     = 8
LR         = 1e-3
N_MFCC     = 40
MAX_LEN    = 100
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

print("Loading dataset...")
raw = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")

def simple_mfcc(audio, sr=16000, n_fft=512, hop=160):
    audio  = np.append(audio[0], audio[1:] - 0.97 * audio[:-1]).astype(np.float32)
    frames = []
    for i in range(0, len(audio) - n_fft, hop):
        frames.append(audio[i:i+n_fft] * np.hamming(n_fft))
    if not frames:
        frames = [np.zeros(n_fft)]
    frames  = np.stack(frames)
    ps      = (1/n_fft) * np.abs(np.fft.rfft(frames, n=n_fft)) ** 2
    # Simple filterbank
    n_filters = N_MFCC
    fb        = np.zeros((n_filters, ps.shape[1]))
    step      = ps.shape[1] // (n_filters + 1)
    for i in range(n_filters):
        fb[i, i*step:(i+2)*step] = 1.0 / (2*step + 1e-8)
    log_mel = np.log(ps @ fb.T + 1e-8)
    feat    = log_mel[:MAX_LEN].T
    if feat.shape[1] < MAX_LEN:
        feat = np.pad(feat, ((0,0),(0, MAX_LEN - feat.shape[1])))
    return feat[:, :MAX_LEN]

class SpeechDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        audio   = np.array(self.data[idx]["audio"]["array"], dtype=np.float32)
        mfcc    = simple_mfcc(audio)
        # Use speaker_id as label, channel as env proxy
        spk_lbl = hash(self.data[idx].get("speaker_id", str(idx))) % 10
        env_lbl = len(audio) % 5
        return (
            torch.tensor(mfcc,    dtype=torch.float32),
            torch.tensor(spk_lbl, dtype=torch.long),
            torch.tensor(env_lbl, dtype=torch.long),
        )

dataset = SpeechDataset(raw)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
N_SPK   = 10
N_ENV   = 5

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam): ctx.lam = lam; return x.clone()
    @staticmethod
    def backward(ctx, grad):  return -ctx.lam * grad, None

def grad_reverse(x, lam=1.0): return GradReverse.apply(x, lam)

class SpeakerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(N_MFCC, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1),     nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.speaker_head = nn.Linear(128, N_SPK)
        self.env_head     = nn.Linear(128, N_ENV)
    def forward(self, x, lam=1.0):
        z = self.conv(x).squeeze(-1)
        return self.speaker_head(z), self.env_head(grad_reverse(z, lam)), z

class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(N_MFCC, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1),     nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, N_SPK)
    def forward(self, x): return self.fc(self.conv(x).squeeze(-1))

def train_model(model, loader, method="ours"):
    opt    = optim.Adam(model.parameters(), lr=LR)
    ce     = nn.CrossEntropyLoss()
    model.to(DEVICE)
    for epoch in range(EPOCHS):
        total_loss, correct, total = 0, 0, 0
        for mfcc, spk_lbl, env_lbl in loader:
            mfcc, spk_lbl, env_lbl = mfcc.to(DEVICE), spk_lbl.to(DEVICE), env_lbl.to(DEVICE)
            opt.zero_grad()
            if method == "ours":
                lam = min(1.0, epoch / EPOCHS * 2)
                spk_out, env_out, _ = model(mfcc, lam)
                loss = ce(spk_out, spk_lbl) + 0.1 * ce(env_out, env_lbl)
            else:
                spk_out = model(mfcc)
                loss    = ce(spk_out, spk_lbl)
            loss.backward(); opt.step()
            total_loss += loss.item()
            correct    += (spk_out.argmax(1) == spk_lbl).sum().item()
            total      += len(spk_lbl)
        print(f"[{method}] Epoch {epoch+1}/{EPOCHS}  Loss={total_loss/len(loader):.4f}  Acc={100*correct/total:.2f}%")
    return model

print("\n--- Training Baseline ---")
baseline = train_model(BaselineCNN(), loader, "baseline")
print("\n--- Training Disentangled ---")
ours     = train_model(SpeakerEncoder(), loader, "ours")

torch.save(baseline.state_dict(), "baseline.pt")
torch.save(ours.state_dict(),     "disentangled.pt")
print("\nSaved: baseline.pt, disentangled.pt")
