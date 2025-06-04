import argparse
import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, recall_score, confusion_matrix, accuracy_score, hamming_loss
from torch.utils.data import DataLoader
from model import ESMAMP
from data import AMPDataset

def evaluate(model, dataloader, label_list, device):
    model.eval()
    all_probs, all_trues = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].cpu().numpy()
            logits = model(input_ids).cpu().numpy()
            probs = 1 / (1 + np.exp(-logits))  # sigmoid

            all_probs.append(probs)
            all_trues.append(targets)

    y_prob = np.vstack(all_probs)
    y_true = np.vstack(all_trues)
    y_pred = (y_prob > 0.5).astype(int)

    mask = (y_true != -1)
    y_true_masked = np.where(mask, y_true, 0)

    results = []
    valid_cols = np.any(mask, axis=0)
    y_true_eval = y_true_masked[:, valid_cols]
    y_pred_eval = y_pred[:, valid_cols]
    y_prob_eval = y_prob[:, valid_cols]
    label_list_eval = np.array(label_list)[valid_cols]

    for i, label in enumerate(label_list_eval):
        true = y_true_eval[:, i]
        pred = y_pred_eval[:, i]
        prob = y_prob_eval[:, i]

        acc = accuracy_score(true, pred)
        f1 = f1_score(true, pred, zero_division=0)
        auc = roc_auc_score(true, prob) if len(np.unique(true)) > 1 else 0.0
        recall = recall_score(true, pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(true, pred).ravel() if len(np.unique(true)) == 2 else (0, 0, 0, 0)
        specificity = tn / (tn + fp + 1e-6) if (tn + fp) > 0 else 0.0

        results.append({
            "Label": label,
            "Acc": acc,
            "F1": f1,
            "AUC": auc,
            "Sensitivity": recall,
            "Specificity": specificity
        })

    macro_f1 = f1_score(y_true_eval, y_pred_eval, average='macro', zero_division=0)
    macro_auc = roc_auc_score(y_true_eval, y_prob_eval, average='macro')
    macro_acc = accuracy_score(y_true_eval.flatten(), y_pred_eval.flatten())
    hamming = hamming_loss(y_true_eval, y_pred_eval)

    return results, {
        "Macro-F1": macro_f1,
        "Macro-AUC": macro_auc,
        "Macro-Acc": macro_acc,
        "Hamming Loss": hamming
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate MTL model")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    label_list = [
        'anti_mammalian_cells', 'antibacterial', 'antibiofilm', 'anticancer', 'antifungal',
        'antigram-negative', 'antigram-positive', 'antihiv', 'antimrsa', 'antioxidant',
        'antiparasitic', 'antiviral', 'cytotoxic', 'hemolytic'
    ]

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    df = pd.read_csv(args.data_path)
    dataset = AMPDataset(df, task_label=label_list)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=dataset.collate_fn, num_workers=0)

    model = ESMAMP(num_labels=len(label_list)).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    per_label_results, overall_metrics = evaluate(model, dataloader, label_list, device)

    print("\n===== Per-label Results =====")
    for r in per_label_results:
        print(f"{r['Label']:<20} | Acc: {r['Acc']:.4f} | F1: {r['F1']:.4f} | AUC: {r['AUC']:.4f} | "
              f"Sens: {r['Sensitivity']:.4f} | Spec: {r['Specificity']:.4f}")

    print("\n===== Overall Metrics =====")
    for k, v in overall_metrics.items():
        print(f"{k:<15}: {v:.4f}")

if __name__ == "__main__":
    main()
