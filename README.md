




选择语言
SCALE-AMP: Multi-Task Learning Model for Antimicrobial Peptide Prediction
This repository implements a deep learning framework called **SCALE-AMP** for both AMP (Antimicrobial Peptide) binary classification and multi-label functional prediction. The model leverages pre-trained ESM-2 embeddings and incorporates several advanced modules including SCConv, ProMamba, and LiteMLA.

📁 Project Structure
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
1
SCALE-AMP/
2
│
3
├── Module/                  # Custom model modules
4
│   ├── litemla.py          # LiteMLA: Multi-scale linear attention module
5
│   ├── ProMamba.py         # ProMambaBlock: protein state modeling
6
│   ├── ScConv.py           # SCConv: local feature reconstruction module
7
│
8
├── model.py                # Definition of SCALE-AMP model
9
├── loss.py                 # Custom loss functions: DualTaskLoss, CASL
10
├── data.py                 # AMPDataset: data loader for AMP/MTL tasks
11
│
12
├── train_amp.py            # Training script for AMP binary classification
13
├── train_mtl.py            # Training script for multi-label classification
⚙️ Environment and Dependencies
Recommended environment: Python 3.8+ and PyTorch 1.12+.

Install the required packages:

1
pip install torch torchvision torchaudio
2
pip install pandas scikit-learn tqdm
3
pip install esm   # Pretrained ESM-2 from Facebook AI
4
pip install iterative-stratification
🚀 How to Run
1. AMP Binary Classification
1
python train_amp.py \
2
  --data data/Benchmark/Stage-1/AMP.csv \
3
  --output_dir checkpoints_amp \
4
  --epochs 100 \
5
  --batch_size 16 \
6
  --lr 5e-5 \
7
  --gpu 0 \
8
  --folds 5
2. Multi-Label Functional Prediction
1
python train_mtl.py \
2
  --data data/Benchmark/Stage-2/MTL.csv \
3
  --output_dir checkpoints_mtl \
4
  --epochs 100 \
5
  --batch_size 16 \
6
  --lr 5e-5 \
7
  --gpu 0 \
8
  --folds 5
📊 Outputs
During training, the following metrics will be reported:

Loss

Macro-F1 score

Macro-AUC score

Hamming Loss

Per-label F1 and AUC (in MTL task)

AMP mixed precision training and early stopping are supported.

For questions or suggestions, feel free to open an issue or contact the author.

2027 字
