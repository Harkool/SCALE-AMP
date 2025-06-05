from typing import Union, Dict
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
import esm
import re


class AMPDataset(Dataset):
    def __init__(
        self,
        data_file: Union[str, Path, pd.DataFrame],
        task_label: Union[str, list] = "AMP",
        max_pep_len: int = 700,
        esm_model=None,
        batch_converter=None,
    ):
        if isinstance(data_file, pd.DataFrame):
            data = data_file.copy()
        else:
            data = pd.read_csv(data_file)

        data.columns = [col.strip().lower() for col in data.columns]
        assert 'sequence' in data.columns, "Input data must contain a 'sequence' column."

        def is_valid(seq):
            return bool(re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWYBXZOU\-]*", seq, flags=re.IGNORECASE))
        data = data[data['sequence'].apply(is_valid)].reset_index(drop=True)

        sequences = data['sequence'].apply(lambda x: x[:max_pep_len])
        self.sequences = sequences.tolist()

        if isinstance(task_label, str) and task_label.lower() == "amp":
            label_col = 'label' if 'label' in data.columns else 'amp'
            labels = data[label_col].astype(int).values.reshape(-1, 1)
        else:
            labels = data[task_label].astype(float).values
        self.targets = labels

        # ✅ 使用最大模型 esm2_t48_15B_UR50D
        if esm_model is None or batch_converter is None:
            esm_model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
            batch_converter = alphabet.get_batch_converter()
        self.esm_model = esm_model.eval()
        if torch.cuda.is_available():
            self.esm_model = self.esm_model.cuda()
        self.batch_converter = batch_converter

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        seq = self.sequences[index]
        return {
            'sequence': seq,
            'target': self.targets[index]
        }

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        sequences = [(str(i), item['sequence']) for i, item in enumerate(batch)]
        labels = np.array([item['target'] for item in batch])

        _, _, tokens = self.batch_converter(sequences)
        device = next(self.esm_model.parameters()).device
        tokens = tokens.to(device)

        with torch.no_grad():
            representations = self.esm_model(tokens, repr_layers=[48])['representations'][48]

        return {
            'input_ids': representations,                  
            'input_mask': torch.ones(representations.shape[:2], device=device), 
            'targets': torch.tensor(labels, dtype=torch.float, device=device),  
        }
