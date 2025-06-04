
# SCALE-AMP: Multi-Scale Attention Model for Functional Antimicrobial Peptide Prediction

**SCALE-AMP** is a deep learning framework designed for both binary classification of antimicrobial peptides (AMPs) and multi-label functional activity prediction. It combines the power of pre-trained protein embeddings from ESM-2 with advanced modules like SCConv (local feature extraction), ProMamba (sequential modeling), and LiteMLA (multi-scale linear attention).

---

## 🧠 Model Architecture

The overall architecture of SCALE-AMP integrates sequence embeddings and multi-scale attention modules to capture both global and local patterns:

```
Input Sequence (FASTA)
       ↓
  ESM-2 Embeddings
       ↓
+---------------------+
| SCConv (Local)      |
| ProMamba (State)    |
| LiteMLA (Attention) |
+---------------------+
       ↓
Shared Feature Representation
       ↓
┌──────────────┬──────────────┐
│ Binary Head  │ Multi-label │
│ (AMP vs Non) │  Heads (k)  │
└──────────────┴──────────────┘
```

> A high-resolution version of this figure can be added in `assets/scale-amp-architecture.png`.

---

## 📁 Project Structure

```
SCALE-AMP/
│
├── Module/                  # Custom model components
│   ├── litemla.py          # LiteMLA: Multi-scale linear attention
│   ├── ProMamba.py         # ProMambaBlock: protein state modeling
│   ├── ScConv.py           # SCConv: local feature extractors
│
├── model.py                # Full SCALE-AMP model
├── loss.py                 # Loss functions: WBCE, CASL, dual-task
├── data.py                 # AMPDataset: dataset wrapper
│
├── train_amp.py            # Binary classification (AMP vs Non-AMP)
├── train_mtl.py            # Multi-label classification (functional tasks)
```

---

## ⚙️ Environment and Dependencies

Install the required packages:

```bash
pip install torch torchvision torchaudio
pip install pandas scikit-learn tqdm
pip install esm   # Pretrained ESM-2 from Facebook AI
pip install iterative-stratification
```

Python ≥3.8 and PyTorch ≥1.12 are recommended.

---

## 🚀 Usage

### Binary Classification (AMP vs Non-AMP)

```bash
python train_amp.py \
  --data data/Benchmark/Stage-1/AMP.csv \
  --output_dir checkpoints_amp \
  --epochs 100 \
  --batch_size 16 \
  --lr 5e-5 \
  --gpu 0 \
  --folds 5
```

### Multi-Label Functional Classification

```bash
python train_mtl.py \
  --data data/Benchmark/Stage-2/MTL.csv \
  --output_dir checkpoints_mtl \
  --epochs 100 \
  --batch_size 16 \
  --lr 5e-5 \
  --gpu 0 \
  --folds 5
```

---

## 📊 Benchmark Results

| Model       | Macro-F1 | Macro-AUC | Hamming Loss |
|-------------|----------|-----------|---------------|
| SCALE-AMP   | 0.473    | 0.826     | 0.124         |
| Deep-Amppred| 0.440    | 0.798     | 0.136         |

> Results based on 5-fold cross-validation on the benchmark dataset.

---

## 📥 Example Input / Output

**Input:**
```
MKWVTFISLLFLFSSAYS
```

**Predicted Output:**
```
[Antibacterial: 0.91, Antifungal: 0.73, Anticancer: 0.12, ...]
```

---

## 📚 Citation

If you use SCALE-AMP in your research, please cite:

```bibtex
@article{your2025scaleamp,
  title={SCALE-AMP: Multi-Scale Attention Model for Functional Antimicrobial Peptide Prediction},
  author={Your Name and Collaborators},
  journal={Bioinformatics},
  year={2025}
}
```

---

For issues, please contact the author or submit a GitHub issue.
