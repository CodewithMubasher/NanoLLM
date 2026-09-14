"""
NanoLLM v2 - Dataset
Data loading, cleaning, packing, and train/val split for language model pretraining.
"""
import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import load_tokenizer, PAD_ID, EOS_ID


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: list[dict], path: str):
    """Save JSONL file."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


class PretrainingDataset(Dataset):
    """Dataset for causal language model pretraining with packed sequences."""

    def __init__(self, data: list[int], context_length: int):
        self.data = data
        self.context_length = context_length

    def __len__(self):
        return max(0, len(self.data) - self.context_length - 1)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.context_length]
        y = self.data[idx + 1: idx + 1 + self.context_length]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def pack_documents(token_ids_list: list[list[int]], context_length: int) -> list[int]:
    """Pack documents into fixed-length sequences separated by EOS."""
    packed = []
    for doc_ids in token_ids_list:
        packed.extend(doc_ids)
        packed.append(EOS_ID)
    return packed


def create_dataloaders(
    packed_data: list[int],
    context_length: int,
    batch_size: int,
    val_ratio: float = 0.05,
    num_workers: int = 0,
):
    """Create train and validation dataloaders."""
    split_idx = int(len(packed_data) * (1 - val_ratio))
    train_data = packed_data[:split_idx]
    val_data = packed_data[split_idx:]

    train_dataset = PretrainingDataset(train_data, context_length)
    val_dataset = PretrainingDataset(val_data, context_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, len(train_data), len(val_data)


def get_batch(loader, device):
    """Get a batch from loader and move to device."""
    x, y = next(iter(loader))
    return x.to(device), y.to(device)
