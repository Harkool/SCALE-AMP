# SCALE-AMP: Multi-Task Learning Model for Antimicrobial Peptide Prediction

This repository implements a deep learning framework called **SCALE-AMP** for both AMP (Antimicrobial Peptide) binary classification and multi-label functional prediction. The model leverages pre-trained ESM-2 embeddings and incorporates several advanced modules including SCConv, ProMamba, and LiteMLA.

---

## 📁 Project Structure

```
SCALE-AMP/
│
├── Module/                  # Custom model modules
│   ├── litemla.py          # LiteMLA: Multi-scale linear attention module
│   ├── ProMamba.py         # ProMambaBlock: protein state modeling
│   ├── ScConv.py           # SCConv: local feature reconstruction module
│
├── model.py                # Definition of SCALE-AMP model
├── loss.py                 # Custom loss functions: DualTaskLoss, CASL
├── data.py                 # AMPDataset: data loader for AMP/MTL tasks
│
├── train_amp.py            # Training script for AMP binary classification
├── train_mtl.py            # Training script for multi-label classification
│
├── evaluate_mtl.py         # Evaluation script for multi-label task
├── evaluate_amp.py         # (Optional) Evaluation for binary task
```

---

## ⚙️ Environment and Dependencies

Recommended environment: Python 3.8+ and PyTorch 1.12+

Install the required packages:

```bash
pip install torch torchvision torchaudio
pip install pandas scikit-learn tqdm
pip install esm  # Pretrained ESM-2 from Facebook AI
pip install iterative-stratification
```

---

## 🚀 How to Run

### 1. AMP Binary Classification

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

### 2. Multi-Label Functional Prediction

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

## 📊 Outputs During Training

Each training run reports the following metrics:

* **Loss**
* **Macro-F1 Score**
* **Macro-AUC Score**
* **Hamming Loss**
* *(MTL only)* Per-label **F1** and **AUC** for all 14 functional tags

AMP training supports mixed precision and early stopping for efficiency and stability.

---

## ✅ Evaluation Scripts


### 1. AMP Binary Evaluation (`evaluate_amp.py`)

Evaluate a binary AMP classification model using:

```bash
python evaluate_amp.py \
  --model_path checkpoints_amp/fold1_best.pt \
  --data_path data/Benchmark/Stage-1/AMP.csv \
  --gpu 0

### 2. Multi-Label Evaluation (`evaluate_mtl.py`)

After training, you can evaluate a saved model checkpoint using:

```bash
python evaluate_mtl.py \
  --model_path checkpoints_mtl/fold1_best.pt \
  --data_path data/Benchmark/Stage-2/MTL.csv \
  --gpu 0
```

This script reports:

* **Per-label metrics** for each of the 14 functional activities:

  * Accuracy, F1 Score, ROC AUC, Sensitivity (Recall), Specificity
* **Overall metrics**:

  * Macro-F1, Macro-AUC, Macro-Accuracy, Hamming Loss

> You can easily extend this script to evaluate all folds or generate ROC curves if needed.

---

## 📬 Contact

For questions, issues, or contributions, feel free to open an issue or reach out to the maintainer.

---

*This project is designed for robust and interpretable AMP discovery, supporting both classification and functional annotation in a unified framework.*
