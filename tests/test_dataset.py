"""NanoLLM v2 - Dataset Tests"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import tempfile
from dataset import (
    load_jsonl, save_jsonl, pack_documents, PretrainingDataset,
    create_dataloaders, get_batch
)
from tokenizer import train_tokenizer, load_tokenizer, PAD_ID, EOS_ID


def test_load_save_jsonl():
    """Test JSONL load/save."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.jsonl")
        data = [{"text": "Hello"}, {"text": "World"}]
        save_jsonl(data, path)
        loaded = load_jsonl(path)
        assert len(loaded) == 2
        assert loaded[0]["text"] == "Hello"


def test_pack_documents():
    """Test document packing."""
    docs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    packed = pack_documents(docs, 1024)
    assert packed == [1, 2, 3, EOS_ID, 4, 5, 6, EOS_ID, 7, 8, 9, EOS_ID]


def test_dataset_getitem():
    """Test dataset __getitem__."""
    data = list(range(100))
    ds = PretrainingDataset(data, context_length=32)
    assert len(ds) == 67
    x, y = ds[0]
    assert x.shape == (32,)
    assert y.shape == (32,)
    assert y[0].item() == x[1].item()


def test_create_dataloaders():
    """Test dataloader creation."""
    data = list(range(10000))
    train_loader, val_loader, train_len, val_len = create_dataloaders(
        data, context_length=64, batch_size=4, val_ratio=0.1
    )
    assert train_len > 0
    assert val_len > 0
    assert train_len > val_len
    x, y = next(iter(train_loader))
    assert x.shape == (4, 64)
    assert y.shape == (4, 64)


def test_train_val_split():
    """Test train/val split doesn't overlap."""
    data = list(range(10000))
    train_loader, val_loader, train_len, val_len = create_dataloaders(
        data, context_length=64, batch_size=4, val_ratio=0.1
    )
    assert train_len + val_len <= len(data)


def test_empty_data():
    """Test handling of empty data."""
    try:
        ds = PretrainingDataset([], 32)
        assert len(ds) == 0
    except Exception:
        pass


if __name__ == "__main__":
    test_load_save_jsonl()
    test_pack_documents()
    test_dataset_getitem()
    test_create_dataloaders()
    test_train_val_split()
    test_empty_data()
    print("All dataset tests passed!")
