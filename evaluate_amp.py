import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, confusion_matrix
from torch.utils.data import DataLoader
from model import ESMAMP
from data import AMPDataset

def evaluate(model, dataloader, device):
    model.eval()
    all_probs, all_preds, all_trues = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'][:, 0].cpu().numpy()

            logits = model(input_ids)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_trues.extend(targets)

    y_true = np.array(all_trues)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    recall = recall_score(y_true, y_pred)  # Sensitivity

    # Specificity: TN / (TN + FP)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp + 1e-6)  

    print("\n====== Evaluation Metrics ======")
    print(f"Accuracy     : {acc:.4f}")
    print(f"F1 Score     : {f1:.4f}")
    print(f"ROC AUC      : {auc:.4f}")
    print(f"Sensitivity  : {recall:.4f}")
    print(f"Specificity  : {specificity:.4f}")
    print(f"Confusion Matrix:\n[[TN: {tn}, FP: {fp}]\n [FN: {fn}, TP: {tp}]]")

def main():
    parser = argparse.ArgumentParser(description="Evaluate AMP model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pt model file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to .csv data file")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load data
    df = pd.read_csv(args.data_path)
    dataset = AMPDataset(df.reset_index(drop=True), task_label="AMP")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=dataset.collate_fn, num_workers=0, pin_memory=False)

    # Load model
    model = ESMAMP(num_labels=1).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    evaluate(model, dataloader, device)

if __name__ == "__main__":
    main()
