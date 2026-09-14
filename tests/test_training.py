"""NanoLLM v2 - Training Tests"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import tempfile
import torch
from config import Config, ModelConfig, TrainConfig
from model import NanoLLM
from train import save_checkpoint, load_checkpoint, get_lr


def test_checkpoint_save_load():
    """Test checkpoint save and load."""
    with tempfile.TemporaryDirectory() as tmp:
        config = Config()
        config.model.num_layers = 2
        config.model.hidden_size = 128
        config.model.num_heads = 2
        config.model.head_dim = 64
        config.model.ffn_hidden_size = 256
        config.model.vocab_size = 1000
        model = NanoLLM(config.model)
        optimizer = model.configure_optimizers(0.1, 6e-4, (0.9, 0.95), "cpu")

        path = os.path.join(tmp, "ckpt.pt")
        save_checkpoint(model, optimizer, None, None, 100, 2.5, config, path)
        assert os.path.exists(path)

        model2 = NanoLLM(config.model)
        opt2 = model2.configure_optimizers(0.1, 6e-4, (0.9, 0.95), "cpu")
        step, best_loss = load_checkpoint(path, model2, opt2)
        assert step == 100
        assert best_loss == 2.5

        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)


def test_resume():
    """Test training resume from checkpoint."""
    with tempfile.TemporaryDirectory() as tmp:
        config = Config()
        config.model.num_layers = 2
        config.model.hidden_size = 128
        config.model.num_heads = 2
        config.model.head_dim = 64
        config.model.ffn_hidden_size = 256
        config.model.vocab_size = 1000
        model = NanoLLM(config.model)
        optimizer = model.configure_optimizers(0.1, 6e-4, (0.9, 0.95), "cpu")

        x = torch.randint(0, 1000, (2, 32))
        _, loss = model(x, x)
        loss.backward()
        optimizer.step()

        path = os.path.join(tmp, "ckpt.pt")
        save_checkpoint(model, optimizer, None, None, 50, 3.0, config, path)

        model2 = NanoLLM(config.model)
        opt2 = model2.configure_optimizers(0.1, 6e-4, (0.9, 0.95), "cpu")
        step, _ = load_checkpoint(path, model2, opt2)
        assert step == 50


def test_lr_schedule():
    """Test learning rate schedule."""
    lr = get_lr(0, warmup_steps=100, max_steps=1000, learning_rate=1e-3, min_lr=1e-5)
    assert lr == 0.0

    lr = get_lr(50, warmup_steps=100, max_steps=1000, learning_rate=1e-3, min_lr=1e-5)
    assert 0 < lr < 1e-3

    lr = get_lr(100, warmup_steps=100, max_steps=1000, learning_rate=1e-3, min_lr=1e-5)
    assert lr == 1e-3

    lr = get_lr(500, warmup_steps=100, max_steps=1000, learning_rate=1e-3, min_lr=1e-5)
    assert 1e-5 < lr < 1e-3

    lr = get_lr(1000, warmup_steps=100, max_steps=1000, learning_rate=1e-3, min_lr=1e-5)
    assert lr == 1e-5


def test_config_serialize():
    """Test config serialization."""
    config = Config()
    d = {"model": config.model.__dict__, "train": config.train.__dict__}
    s = json.dumps(d)
    d2 = json.loads(s)
    assert d2["model"]["vocab_size"] == config.model.vocab_size


if __name__ == "__main__":
    test_checkpoint_save_load()
    test_resume()
    test_lr_schedule()
    test_config_serialize()
    print("All training tests passed!")
