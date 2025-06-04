import os, argparse
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from data import AMPDataset
from model import ESMAMP
from torch.nn import BCEWithLogitsLoss
from torch.cuda.amp import autocast, GradScaler


def train(dataloader, model, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch}")):
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)

        if targets.dim() == 1:
            targets = targets.view(-1, 1)

        with autocast():
            logits = model(input_ids)
            if logits.dim() != targets.dim():
                targets = targets.view_as(logits)
            loss = criterion(logits, targets)

        if torch.isnan(loss):
            print(f"❌ [Epoch {epoch} | Step {step}] Loss is NaN! Skipping batch.")
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
    all_probs, all_targets = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating Epoch {epoch}"):
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'][:, 0].cpu().numpy()
            logits = model(input_ids)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

            all_probs.append(probs)
            all_targets.append(targets)

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    valid_mask = ~np.isnan(y_prob)
    y_prob = y_prob[valid_mask]
    y_true = y_true[valid_mask]

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        acc, f1, auc = 0.0, 0.0, 0.0
    else:
        y_pred = (y_prob > 0.5).astype(int)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)

    return acc, f1, auc


def main():
    parser = argparse.ArgumentParser(description="Train AMP Classifier with 5-fold CV")
    parser.add_argument("--data", type=str, default="data/Benchmark/AMP.csv")
    parser.add_argument("--output_dir", type=str, default="checkpoints_amp_cv")
    parser.add_argument("--epochs", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.data)
    labels = df['label'].values.astype(int)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, labels)):
        print(f"\n========== Fold {fold + 1}/{args.folds} ==========")

        train_set = AMPDataset(df.iloc[train_idx].reset_index(drop=True), task_label="AMP")
        val_set = AMPDataset(df.iloc[val_idx].reset_index(drop=True), task_label="AMP")

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=train_set.collate_fn, num_workers=0, pin_memory=False)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                collate_fn=val_set.collate_fn, num_workers=0, pin_memory=False)

        model = ESMAMP(num_labels=1).to(device)
        criterion = BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        scaler = GradScaler()

        best_f1 = 0.0
        for epoch in range(1, args.epochs + 1):
            loss = train(train_loader, model, criterion, optimizer, scaler, device, epoch)
            acc, f1, auc = evaluate(val_loader, model, device, epoch)
            scheduler.step()

            print(f"[Fold {fold+1} | Epoch {epoch}] Loss: {loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                ckpt_path = os.path.join(args.output_dir, f"fold{fold+1}_best.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"✅ Saved best model to {ckpt_path}")


if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
