"""
Q3 - Fair Training with Fairness Loss
Dataset: hf-internal-testing/librispeech_asr_demo
Simulates demographic groups for fairness demonstration
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS     = 8
N_MFCC     = 40
MAX_LEN    = 100
print(f"Device: {DEVICE}")

def simple_mfcc(audio, n_fft=512, hop=160):
    audio  = np.append(audio[0], audio[1:] - 0.97*audio[:-1]).astype(np.float32)
    frames = []
    for i in range(0, max(1, len(audio)-n_fft), hop):
        frames.append(audio[i:i+n_fft] * np.hamming(n_fft))
    if not frames: frames = [np.zeros(n_fft)]
    frames  = np.stack(frames)
    ps      = (1/n_fft) * np.abs(np.fft.rfft(frames, n=n_fft)) ** 2
    fb      = np.zeros((N_MFCC, ps.shape[1]))
    step    = ps.shape[1] // (N_MFCC + 1)
    for i in range(N_MFCC):
        fb[i, i*step:(i+2)*step] = 1.0 / (2*step + 1e-8)
    log_mel = np.log(ps @ fb.T + 1e-8)
    feat    = log_mel[:MAX_LEN].T
    if feat.shape[1] < MAX_LEN:
        feat = np.pad(feat, ((0,0),(0, MAX_LEN-feat.shape[1])))
    return feat[:, :MAX_LEN]

print("Loading dataset...")
raw = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")

class SpeechDataset(Dataset):
    def __init__(self, data):
        self.data = data
        np.random.seed(42)
        # Simulate gender groups: 0=male, 1=female, 2=other
        self.groups = np.random.choice([0,1,2], size=len(data), p=[0.6, 0.3, 0.1])
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        audio = np.array(self.data[idx]["audio"]["array"], dtype=np.float32)
        mfcc  = simple_mfcc(audio)
        label = int(np.mean(audio**2) > 1e-4)   # speech vs silence
        return (torch.tensor(mfcc,             dtype=torch.float32),
                torch.tensor(label,            dtype=torch.long),
                torch.tensor(self.groups[idx], dtype=torch.long))

loader = DataLoader(SpeechDataset(raw), batch_size=BATCH_SIZE, shuffle=True)

class SimpleASR(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(N_MFCC, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(64, 2)
    def forward(self, x): return self.fc(self.net(x).squeeze(-1))

class FairnessLoss(nn.Module):
    """Minimize variance of per-group loss = equalize performance across groups"""
    def __init__(self, n_groups=3, lam=0.5):
        super().__init__()
        self.n_groups = n_groups
        self.lam      = lam
        self.ce       = nn.CrossEntropyLoss(reduction="none")
    def forward(self, logits, labels, groups):
        per_sample = self.ce(logits, labels)
        base_loss  = per_sample.mean()
        group_losses = []
        for g in range(self.n_groups):
            mask = (groups == g)
            if mask.sum() > 0:
                group_losses.append(per_sample[mask].mean())
        if len(group_losses) > 1:
            fair_loss = torch.var(torch.stack(group_losses))
        else:
            fair_loss = torch.tensor(0.0)
        return base_loss + self.lam * fair_loss, base_loss.item(), fair_loss.item()

model     = SimpleASR().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = FairnessLoss(n_groups=3, lam=0.5)

print("\n--- Training with Fairness Loss ---")
history = {"loss": [], "base": [], "fair": [], "acc": []}

for epoch in range(EPOCHS):
    total_loss, total_base, total_fair = 0, 0, 0
    correct, total = 0, 0
    for mfcc, lbl, grp in loader:
        mfcc, lbl, grp = mfcc.to(DEVICE), lbl.to(DEVICE), grp.to(DEVICE)
        optimizer.zero_grad()
        out = model(mfcc)
        loss, base, fair = criterion(out, lbl, grp)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_base += base
        total_fair += fair
        correct    += (out.argmax(1) == lbl).sum().item()
        total      += len(lbl)
    n   = len(loader)
    acc = 100 * correct / total
    history["loss"].append(total_loss/n)
    history["base"].append(total_base/n)
    history["fair"].append(total_fair/n)
    history["acc"] .append(acc)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss={total_loss/n:.4f} | Base={total_base/n:.4f} | Fair={total_fair/n:.4f} | Acc={acc:.1f}%")

torch.save(model.state_dict(), "fair_model.pt")

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
epochs = range(1, EPOCHS+1)
axes[0].plot(epochs, history["loss"], label="Total Loss", color="red")
axes[0].plot(epochs, history["base"], label="Base CE Loss", color="blue", linestyle="--")
axes[0].plot(epochs, history["fair"], label="Fairness Loss", color="green", linestyle="--")
axes[0].set_title("Training Loss Curves")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(epochs, history["acc"], color="purple", marker="o")
axes[1].set_title("Training Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].grid(alpha=0.3)

plt.suptitle("Fair Training — Fairness Loss Minimizes Cross-Group Disparity")
plt.tight_layout()
plt.savefig("fair_training_curves.png", dpi=150)
plt.show()
print("Done! Saved → fair_model.pt, fair_training_curves.png")
