"""NanoLLM v2 - Model Tests"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config import ModelConfig
from model import NanoLLM, count_parameters, RMSNorm, CausalSelfAttention, SwiGLU


def test_model_construction():
    """Test model can be built."""
    config = ModelConfig()
    model = NanoLLM(config)
    assert model is not None


def test_parameter_count():
    """Test parameter count is reasonable."""
    config = ModelConfig()
    model = NanoLLM(config)
    n = sum(p.numel() for p in model.parameters())
    assert 40_000_000 < n < 60_000_000, f"Parameter count {n} not in 40M-60M range"


def test_forward_pass():
    """Test forward pass produces correct shapes."""
    config = ModelConfig()
    model = NanoLLM(config)
    x = torch.randint(0, config.vocab_size, (2, 64))
    logits, loss = model(x, x)
    assert logits.shape == (2, 64, config.vocab_size)
    assert loss is not None
    assert torch.isfinite(loss)


def test_forward_no_targets():
    """Test forward pass without targets (inference mode)."""
    config = ModelConfig()
    model = NanoLLM(config)
    x = torch.randint(0, config.vocab_size, (1, 32))
    logits, loss = model(x)
    assert logits.shape == (1, 1, config.vocab_size)
    assert loss is None


def test_causal_mask():
    """Test causal masking prevents attending to future tokens."""
    config = ModelConfig()
    model = NanoLLM(config)
    model.eval()
    x = torch.randint(0, config.vocab_size, (1, 64))
    logits1, _ = model(x, x)
    x2 = x.clone()
    x2[:, 32:] = torch.randint(0, config.vocab_size, (1, 32))
    logits2, _ = model(x2, x2)
    assert not torch.allclose(logits1, logits2), "Causal mask not working - outputs are identical"


def test_generate():
    """Test generation produces valid tokens."""
    config = ModelConfig()
    model = NanoLLM(config)
    model.eval()
    x = torch.randint(0, config.vocab_size, (1, 10))
    out = model.generate(x, max_new_tokens=20, temperature=1.0, top_k=50)
    assert out.shape == (1, 30)
    assert (out[:, :10] == x).all()


def test_loss_decreases():
    """Test loss decreases with a few gradient steps."""
    config = ModelConfig()
    config.num_layers = 2
    config.hidden_size = 128
    config.num_heads = 2
    config.head_dim = 64
    config.ffn_hidden_size = 256
    model = NanoLLM(config)
    x = torch.randint(0, config.vocab_size, (4, 32))

    losses = []
    for _ in range(10):
        _, loss = model(x, x)
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for p in model.parameters():
            if p.grad is not None:
                p.data -= 0.01 * p.grad
            p.grad = None

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_rmsnorm():
    """Test RMSNorm."""
    norm = RMSNorm(64)
    x = torch.randn(2, 10, 64)
    out = norm(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_swiglu():
    """Test SwiGLU."""
    from config import ModelConfig
    config = ModelConfig()
    ffn = SwiGLU(config)
    x = torch.randn(2, 10, config.hidden_size)
    out = ffn(x)
    assert out.shape == x.shape


if __name__ == "__main__":
    test_model_construction()
    test_parameter_count()
    test_forward_pass()
    test_forward_no_targets()
    test_causal_mask()
    test_generate()
    test_loss_decreases()
    test_rmsnorm()
    test_swiglu()
    print("All model tests passed!")
