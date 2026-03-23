"""
Q3 - Bias Audit
Dataset: speech_commands via librispeech_asr_demo — no login needed
We simulate demographic metadata for audit demonstration
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from datasets import load_dataset

print("Loading dataset...")
ds = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")

# Simulate demographic metadata (as Common Voice would have)
# In real scenario this comes from dataset metadata fields
np.random.seed(42)
n = len(ds)

GENDERS  = ["male", "female", "other", "unknown"]
AGES     = ["teens", "twenties", "thirties", "fourties", "fifties", "unknown"]
ACCENTS  = ["us", "england", "australia", "unknown"]

# Simulate realistic bias — more male, more unknown age (documentation debt)
gender_probs = [0.55, 0.20, 0.05, 0.20]   # male-heavy
age_probs    = [0.05, 0.15, 0.10, 0.08, 0.02, 0.60]  # mostly unknown
accent_probs = [0.50, 0.10, 0.05, 0.35]   # US-heavy

records = []
for i in range(n):
    records.append({
        "gender": np.random.choice(GENDERS,  p=gender_probs),
        "age":    np.random.choice(AGES,     p=age_probs),
        "accent": np.random.choice(ACCENTS,  p=accent_probs),
        "duration": len(ds[i]["audio"]["array"]) / ds[i]["audio"]["sampling_rate"]
    })

print(f"Auditing {len(records)} samples\n")

# ── Documentation Debt ────────────────────
def doc_debt(records, field):
    missing = sum(1 for r in records if r[field] == "unknown")
    return missing, 100 * missing / len(records)

print("=" * 45)
print("DOCUMENTATION DEBT ANALYSIS")
print("=" * 45)
debt_results = {}
for field in ["gender", "age", "accent"]:
    n_miss, pct = doc_debt(records, field)
    debt_results[field] = pct
    print(f"  {field:<10}: {n_miss:>4} missing  ({pct:.1f}%)")

# ── Representation Bias ───────────────────
def count_field(records, field):
    vals = [r[field] for r in records if r[field] != "unknown"]
    return Counter(vals)

gender_counts = count_field(records, "gender")
age_counts    = count_field(records, "age")
accent_counts = count_field(records, "accent")

print("\nGENDER DISTRIBUTION (labelled only):")
for k, v in sorted(gender_counts.items(), key=lambda x: -x[1]):
    bar = "█" * int(v * 30 / max(gender_counts.values()))
    print(f"  {k:<12}: {v:>4}  {bar}")

print("\nAGE DISTRIBUTION (labelled only):")
for k, v in sorted(age_counts.items(), key=lambda x: -x[1]):
    bar = "█" * int(v * 30 / max(age_counts.values()))
    print(f"  {k:<12}: {v:>4}  {bar}")

print("\nACCENT DISTRIBUTION (labelled only):")
for k, v in accent_counts.most_common():
    print(f"  {k:<12}: {v}")

# ── Plot ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Gender pie
if gender_counts:
    axes[0].pie(gender_counts.values(), labels=gender_counts.keys(),
                autopct="%1.1f%%", startangle=90,
                colors=["#3498db", "#e74c3c", "#95a5a6"])
    axes[0].set_title("Gender Distribution\n(labelled samples only)")

# Age bar
if age_counts:
    ages = list(age_counts.keys())
    cnts = list(age_counts.values())
    axes[1].barh(ages, cnts, color="#2ecc71")
    axes[1].set_title("Age Distribution\n(labelled samples only)")
    axes[1].set_xlabel("Count")

# Documentation debt
fields = list(debt_results.keys())
pcts   = list(debt_results.values())
colors = ["#e74c3c" if p > 50 else "#f39c12" for p in pcts]
axes[2].bar(fields, pcts, color=colors)
axes[2].set_title("Documentation Debt (%)\n(missing metadata)")
axes[2].set_ylabel("% Missing / Unknown")
axes[2].set_ylim(0, 100)
axes[2].axhline(50, color="red", linestyle="--", label="50% threshold")
axes[2].legend()
for i, (f, p) in enumerate(zip(fields, pcts)):
    axes[2].text(i, p + 1, f"{p:.1f}%", ha="center", fontsize=10)

plt.suptitle("LibriSpeech Demo — Bias & Documentation Debt Audit", fontsize=13)
plt.tight_layout()
plt.savefig("audit_plots.png", dpi=150)
plt.show()
print("\nDone! Saved → audit_plots.png")
