## Project Structure

    Speech Assignment/
    │
    ├── q1/
    │   ├── mfcc_manual.py
    │   ├── leakage_snr.py
    │   ├── voiced_unvoiced.py
    │   └── phonetic_mapping.py
    │
    ├── q2/
    │   ├── train.py
    │   └── eval.py
    │
    ├── q3/
    │   ├── audit.py
    │   ├── privacymodule.py
    │   └── train_fair.py
    │
    ├── requirements.txt
    └── README.md

---

This assignment explores core concepts in speech processing, including:

- Feature extraction (MFCC)
- Signal analysis (SNR, leakage, voiced/unvoiced)
- Deep learning models for speaker recognition
- Ethical aspects such as fairness, bias, and privacy

The work is divided into three parts:

- Q1: Signal Processing & Feature Extraction  
- Q2: Deep Learning Models  
- Q3: Ethics, Fairness, and Privacy  


---

## Dataset Used

- Dataset: hf-internal-testing/librispeech_asr_demo
- Source: Hugging Face
- Properties:
  - Clean English speech
  - 16 kHz sampling rate
  - Small and lightweight (no authentication required)

---

## Tools & Technologies

- Python  
- PyTorch  
- Hugging Face Transformers  
- NumPy, Matplotlib, Scikit-learn  

---

## Setup Instructions

py -3.10 -m venv venv

venv\Scripts\activate

python -m pip install --upgrade pip

pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu

pip install transformers==4.38.0 datasets==2.19.0 "numpy<2" matplotlib scikit-learn soundfile librosa

---

## How to Run

### Q1 — Signal Processing

python q1/mfcc_manual.py

python q1/leakage_snr.py

python q1/voiced_unvoiced.py

python q1/phonetic_mapping.py


---

### Q2 — Deep Learning

python q2/train.py

python q2/eval.py

---

### Q3 — Ethics & Fairness

python q3/audit.py

python q3/privacymodule.py

python q3/train_fair.py

---

## Detailed Results

### Q1: Signal Processing & Feature Extraction

#### MFCC Computation

MFCCs were implemented manually using:

- Pre-emphasis filtering  
- Framing & windowing  
- FFT  
- Mel filter banks  
- Log scaling  
- DCT  

Output Shape:
(584, 13)

#### Spectral Leakage & SNR

Window | SNR (dB) | Leakage
Rectangular | 16.32 | 0.338
Hamming | 15.00 | 0.429
Hanning | 14.39 | 0.435

Observations:
- Rectangular → higher SNR but more leakage  
- Hamming/Hanning → reduced leakage but lower SNR  

#### Voiced vs Unvoiced Detection

- Voiced frames: 735  
- Unvoiced frames: 512  
- Transitions: 173  

#### Phonetic Boundary Mapping

- Manual boundaries: 173  
- Model boundaries: 290  
- RMSE: 117.12 ms  

---

### Q2: Deep Learning Models

Models:
- Baseline CNN
- Disentangled Representation Model

Results:
- Accuracy: 100% (both models)  
- Loss approaches zero quickly  

Analysis:
- Overfitting due to small dataset  
- Disentangled model shows structured clustering but unstable loss  

---

### Q3: Ethics, Fairness & Privacy

#### Dataset Audit

Attribute | Missing Data
Gender | 15.1%
Age | 54.8%
Accent | 35.6%

#### Bias Observations

- Gender imbalance (male dominant)  
- Accent skewed toward US speakers  
- Limited age diversity  

#### Privacy Module

- Noise addition  
- Pitch shifting  
- Time stretching  
- Signal masking  

#### Fairness Training

L_total = L_CE + λ × Var(L_CE across groups)

Results:
- Accuracy: 100%  
- Fairness loss ≈ 0  

---

## Discussion

- MFCC remains fundamental  
- Deep learning overfits small datasets  
- Bias and missing metadata are common  
- Ethical AI must balance accuracy, fairness, and privacy  

---

## Limitations

- Small dataset leads to overfitting  
- No GPU limits scalability  
- Approximate phonetic alignment  
- Incomplete metadata  

---

## Conclusion

This project demonstrates:

- A complete end-to-end speech processing pipeline 
- Robust feature extraction and signal analysis techniques 
- Implementation of deep learning models 
- Ethical assessment of AI systems 
 
