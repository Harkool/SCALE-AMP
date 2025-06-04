import os, argparse
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, hamming_loss
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from data import AMPDataset
from model import ESMAMP
from loss import MultiLabelCASL
from torch.cuda.amp import autocast, GradScaler

def train(dataloader, model, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0

    for step, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch}")):
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)

        with autocast():
            logits = model(input_ids)
            loss = criterion(logits, targets)

        if torch.isnan(loss):
            print(f"❌ [Epoch {epoch} | Step {step}] NaN Loss, skipping batch.")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss += loss.item()
    return total_loss / max(len(dataloader), 1)

def evaluate(dataloader, model, device, epoch):
    model.eval()
    all_probs, all_trues = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating Epoch {epoch}"):
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)

            logits = model(input_ids)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_trues.append(targets.cpu().numpy())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_trues)

    y_true_masked = np.where(y_true == -1, 0, y_true)
    valid_cols = ~(np.all(y_true == -1, axis=0))
    y_true_eval = y_true_masked[:, valid_cols]
    y_pred = (y_prob > 0.5).astype(int)[:, valid_cols]
    y_prob_eval = y_prob[:, valid_cols]

    macro_f1 = f1_score(y_true_eval, y_pred, average='macro', zero_division=0)
    auc_macro = roc_auc_score(y_true_eval, y_prob_eval, average='macro')
    hamming = hamming_loss(y_true_eval, y_pred)

    return {
        "Macro-F1": macro_f1,
        "AUC": auc_macro,
        "Hamming": hamming
    }

def main():
    parser = argparse.ArgumentParser(description="MTL Functional Classification with 5-fold CV")
    parser.add_argument("--data", type=str, default="data/Benchmark/Stage-2/MTL.csv")
    parser.add_argument("--output_dir", type=str, default="checkpoints_mtl_cv")
    parser.add_argument("--epochs", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    label_list = [
        'anti_mammalian_cells', 'antibacterial', 'antibiofilm', 'anticancer', 'antifungal',
        'antigram-negative', 'antigram-positive', 'antihiv', 'antimrsa', 'antioxidant',
        'antiparasitic', 'antiviral', 'cytotoxic', 'hemolytic'
    ]

    df = pd.read_csv(args.data)
    label_matrix = df[label_list].replace(-1, 0).values
    mskf = MultilabelStratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(mskf.split(np.zeros(len(df)), label_matrix)):
        print(f"\n========== Fold {fold + 1}/{args.folds} ==========")

        train_set = AMPDataset(df.iloc[train_idx].reset_index(drop=True), task_label=label_list)
        val_set = AMPDataset(df.iloc[val_idx].reset_index(drop=True), task_label=label_list)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=train_set.collate_fn, num_workers=0, pin_memory=False)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                collate_fn=val_set.collate_fn, num_workers=0, pin_memory=False)

        model = ESMAMP(num_labels=len(label_list)).to(device)
        criterion = MultiLabelCASL()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        scaler = GradScaler()

        best_f1 = 0.0
        for epoch in range(1, args.epochs + 1):
            loss = train(train_loader, model, criterion, optimizer, scaler, device, epoch)
            perf = evaluate(val_loader, model, device, epoch)
            scheduler.step()

            print(f"[Fold {fold+1} | Epoch {epoch}] Loss: {loss:.4f} | "
                  f"Macro-F1: {perf['Macro-F1']:.4f} | AUC: {perf['AUC']:.4f} | Hamming: {perf['Hamming']:.4f}")

            if perf["Macro-F1"] > best_f1:
                best_f1 = perf["Macro-F1"]
                ckpt_path = os.path.join(args.output_dir, f"fold{fold+1}_best.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"✅ Saved best model to {ckpt_path}")

if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
