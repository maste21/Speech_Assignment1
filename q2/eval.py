"""
Q2 - Evaluation
Dataset: hf-internal-testing/librispeech_asr_demo
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE

N_MFCC, MAX_LEN, N_SPK, N_ENV = 40, 100, 10, 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def simple_mfcc(audio, n_fft=512, hop=160):
    audio  = np.append(audio[0], audio[1:] - 0.97 * audio[:-1]).astype(np.float32)
    frames = []
    for i in range(0, len(audio) - n_fft, hop):
        frames.append(audio[i:i+n_fft] * np.hamming(n_fft))
    if not frames: frames = [np.zeros(n_fft)]
    frames = np.stack(frames)
    ps     = (1/n_fft) * np.abs(np.fft.rfft(frames, n=n_fft)) ** 2
    fb     = np.zeros((N_MFCC, ps.shape[1]))
    step   = ps.shape[1] // (N_MFCC + 1)
    for i in range(N_MFCC):
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
        spk_lbl = hash(self.data[idx].get("speaker_id", str(idx))) % N_SPK
        env_lbl = len(audio) % N_ENV
        return torch.tensor(mfcc, dtype=torch.float32), torch.tensor(spk_lbl, dtype=torch.long), torch.tensor(env_lbl, dtype=torch.long)

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam): ctx.lam = lam; return x.clone()
    @staticmethod
    def backward(ctx, grad):  return -ctx.lam * grad, None
def grad_reverse(x, lam=1.0): return GradReverse.apply(x, lam)

class SpeakerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(N_MFCC,64,3,padding=1),nn.ReLU(),nn.Conv1d(64,128,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
        self.speaker_head = nn.Linear(128, N_SPK)
        self.env_head     = nn.Linear(128, N_ENV)
    def forward(self, x, lam=1.0):
        z = self.conv(x).squeeze(-1)
        return self.speaker_head(z), self.env_head(grad_reverse(z,lam)), z

class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(N_MFCC,64,3,padding=1),nn.ReLU(),nn.Conv1d(64,128,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
        self.fc   = nn.Linear(128, N_SPK)
    def forward(self, x): return self.fc(self.conv(x).squeeze(-1))

print("Loading data...")
raw     = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
loader  = DataLoader(SpeechDataset(raw), batch_size=16, shuffle=False)

baseline = BaselineCNN().to(DEVICE)
ours     = SpeakerEncoder().to(DEVICE)
baseline.load_state_dict(torch.load("baseline.pt",     map_location=DEVICE))
ours    .load_state_dict(torch.load("disentangled.pt", map_location=DEVICE))
baseline.eval(); ours.eval()

def evaluate(model, loader, method):
    correct, total, all_emb, all_lbl = 0, 0, [], []
    with torch.no_grad():
        for mfcc, spk_lbl, _ in loader:
            mfcc, spk_lbl = mfcc.to(DEVICE), spk_lbl.to(DEVICE)
            if method == "ours":
                out, _, emb = model(mfcc)
            else:
                out = model(mfcc)
                emb = model.conv(mfcc).squeeze(-1)
            correct += (out.argmax(1) == spk_lbl).sum().item()
            total   += len(spk_lbl)
            all_emb.append(emb.cpu().numpy())
            all_lbl.append(spk_lbl.cpu().numpy())
    return 100*correct/total, np.vstack(all_emb), np.concatenate(all_lbl)

acc_base, emb_base, lbl_base = evaluate(baseline, loader, "baseline")
acc_ours, emb_ours, lbl_ours = evaluate(ours,     loader, "ours")

print(f"\n{'='*40}")
print(f"{'Baseline CNN':<25} {acc_base:>9.2f}%")
print(f"{'Disentangled':<25} {acc_ours:>9.2f}%")
print("=" * 40)

print("Running t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(emb_base)-1))
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, emb, lbl, title in [
    (axes[0], emb_base, lbl_base, f"Baseline  (Acc={acc_base:.1f}%)"),
    (axes[1], emb_ours, lbl_ours, f"Disentangled (Acc={acc_ours:.1f}%)"),
]:
    proj = tsne.fit_transform(emb)
    ax.scatter(proj[:,0], proj[:,1], c=lbl, cmap="tab10", s=40, alpha=0.8)
    ax.set_title(title); ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
plt.suptitle("Speaker Embedding Space — t-SNE")
plt.tight_layout()
plt.savefig("tsne_comparison.png", dpi=150)
plt.show()
print("Done! Saved → tsne_comparison.png")
