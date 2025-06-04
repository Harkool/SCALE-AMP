# ESMAMP: Multi-Task Learning Model for Antimicrobial Peptide Prediction

This repository implements a deep learning framework called **ESMAMP** for both AMP (Antimicrobial Peptide) binary classification and multi-label functional prediction. The model leverages pre-trained ESM-2 embeddings and incorporates several advanced modules including SCConv, ProMamba, and LiteMLA.

## 📁 Project Structure

```
ESMAMP/
│
├── Module/                  # Custom model modules
│   ├── litemla.py          # LiteMLA: Multi-scale linear attention module
│   ├── ProMamba.py         # ProMambaBlock: protein state modeling
│   ├── ScConv.py           # SCConv: local feature reconstruction module
│
├── model.py                # Definition of ESMAMP model
├── loss.py                 # Custom loss functions: DualTaskLoss, CASL
├── data.py                 # AMPDataset: data loader for AMP/MTL tasks
│
├── train_amp.py            # Training script for AMP binary classification
├── train_mtl.py            # Training script for multi-label classification
```

## ⚙️ Environment and Dependencies

Recommended environment: Python 3.8+ and PyTorch 1.12+.

Install the required packages:

```
pip install torch torchvision torchaudio
pip install pandas scikit-learn tqdm
pip install esm   # Pretrained ESM-2 from Facebook AI
pip install iterative-stratification
```

## 🚀 How to Run

### 1. AMP Binary Classification

```
python train_amp.py \
  --data data/Benchmark/Stage-1/AMP.csv \
  --output_dir checkpoints_amp \
  --epochs 100 \
  --batch_size 16 \
  --lr 5e-5 \
  --gpu 0 \
  --folds 5
```

### 2. Multi-Label Functional Prediction

```
python train_mtl.py \
  --data data/Benchmark/Stage-2/MTL.csv \
  --output_dir checkpoints_mtl \
  --epochs 100 \
  --batch_size 16 \
  --lr 5e-5 \
  --gpu 0 \
  --folds 5
```

## 📊 Outputs

During training, the following metrics will be reported:

* Loss

* Macro-F1 score

* Macro-AUC score

* Hamming Loss

* Per-label F1 and AUC (in MTL task)

AMP mixed precision training and early stopping are supported.

For questions or suggestions, feel free to open an issue or contact the author.
