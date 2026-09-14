"""
NanoLLM v2 - Full Smoke Test
Runs complete pipeline: tokenizer, model, forward, backward, checkpoint, inference.
"""
import sys
import os
import tempfile
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, ModelConfig
from model import NanoLLM, count_parameters
from tokenizer import train_tokenizer, load_tokenizer, encode, decode, tokenize_document, tokenize_documents
from dataset import PretrainingDataset, pack_documents, create_dataloaders
from train import save_checkpoint, load_checkpoint, get_lr, run_preflight

PASS = 0
FAIL = 0
results = []

def test(name, fn):
    global PASS, FAIL
    try:
        fn()
        PASS += 1
        results.append((name, True, ""))
    except Exception as e:
        FAIL += 1
        results.append((name, False, str(e)))


def test_tokenizer():
    with tempfile.TemporaryDirectory() as tmp:
        corpus = os.path.join(tmp, "corpus.txt")
        with open(corpus, "w") as f:
            f.write("The quick brown fox jumps over the lazy dog. ")
            f.write("Python is a programming language. ")
            f.write("Machine learning is a subset of artificial intelligence. ")
            f.write("The capital of France is Paris. ")
            f.write("Water is composed of hydrogen and oxygen. ")
            f.write("Hello world, how are you doing today? ")
            f.write("I am learning to build language models from scratch. ")
            f.write("This is a test sentence for the tokenizer. " * 50)

        path = os.path.join(tmp, "tok.json")
        tok = train_tokenizer([corpus], path, vocab_size=16384)
        loaded = load_tokenizer(path)

        for text in ["Hello world", "How are you", "Python is great", "The sky is blue"]:
            ids = encode(text, loaded, add_special=False)
            decoded = decode(ids, loaded, skip_special=True)
            assert "Ġ" not in decoded, f"BPE marker: {decoded}"
            assert "Ċ" not in decoded, f"BPE marker: {decoded}"

        ids = tokenize_document("Test", loaded)
        assert ids[0] == loaded.token_to_id("<BOS>")
        assert ids[-1] == loaded.token_to_id("<EOS>")


def test_model_build():
    config = ModelConfig()
    model = NanoLLM(config)
    n = sum(p.numel() for p in model.parameters())
    assert 45_000_000 < n < 55_000_000, f"Params {n} not in 45M-55M"


def test_forward_pass():
    config = ModelConfig()
    model = NanoLLM(config)
    x = torch.randint(0, config.vocab_size, (2, 128))
    logits, loss = model(x, x)
    assert logits.shape == (2, 128, config.vocab_size)
    assert torch.isfinite(loss)


def test_backward_pass():
    config = ModelConfig()
    model = NanoLLM(config)
    x = torch.randint(0, config.vocab_size, (2, 64))
    _, loss = model(x, x)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads)


def test_optimizer():
    config = ModelConfig()
    model = NanoLLM(config)
    opt = model.configure_optimizers(0.1, 6e-4, (0.9, 0.95), "cpu")
    x = torch.randint(0, config.vocab_size, (2, 64))
    _, loss = model(x, x)
    loss.backward()
    opt.step()
    opt.zero_grad()


def test_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        config = Config()
        config.model.num_layers = 2
        config.model.hidden_size = 128
        config.model.num_heads = 2
        config.model.head_dim = 64
        config.model.ffn_hidden_size = 256
        config.model.vocab_size = 1000

        model = NanoLLM(config.model)
        opt = model.configure_optimizers(0.1, 6e-4, (0.9, 0.95), "cpu")

        path = os.path.join(tmp, "ckpt.pt")
        save_checkpoint(model, opt, None, None, 100, 2.5, config, path)

        model2 = NanoLLM(config.model)
        opt2 = model2.configure_optimizers(0.1, 6e-4, (0.9, 0.95), "cpu")
        step, best_loss = load_checkpoint(path, model2, opt2)
        assert step == 100
        assert best_loss == 2.5


def test_resume():
    with tempfile.TemporaryDirectory() as tmp:
        config = Config()
        config.model.num_layers = 2
        config.model.hidden_size = 128
        config.model.num_heads = 2
        config.model.head_dim = 64
        config.model.ffn_hidden_size = 256
        config.model.vocab_size = 1000

        model = NanoLLM(config.model)
        opt = model.configure_optimizers(0.1, 6e-4, (0.9, 0.95), "cpu")

        for _ in range(5):
            x = torch.randint(0, 1000, (2, 32))
            _, loss = model(x, x)
            loss.backward()
            opt.step()
            opt.zero_grad()

        path = os.path.join(tmp, "ckpt.pt")
        save_checkpoint(model, opt, None, None, 50, 3.0, config, path)

        model2 = NanoLLM(config.model)
        opt2 = model2.configure_optimizers(0.1, 6e-4, (0.9, 0.95), "cpu")
        step, _ = load_checkpoint(path, model2, opt2)
        assert step == 50


def test_generate():
    config = ModelConfig()
    model = NanoLLM(config)
    model.eval()
    x = torch.randint(0, config.vocab_size, (1, 10))
    out = model.generate(x, max_new_tokens=20, temperature=1.0, top_k=50)
    assert out.shape == (1, 30)


def test_lr_schedule():
    lr0 = get_lr(0, 100, 1000, 1e-3, 1e-5)
    assert lr0 == 0.0
    lr50 = get_lr(50, 100, 1000, 1e-3, 1e-5)
    assert 0 < lr50 < 1e-3
    lr100 = get_lr(100, 100, 1000, 1e-3, 1e-5)
    assert lr100 == 1e-3
    lr_end = get_lr(1000, 100, 1000, 1e-3, 1e-5)
    assert lr_end == 1e-5


def test_dataset():
    data = list(range(10000))
    ds = PretrainingDataset(data, 64)
    assert len(ds) > 0
    x, y = ds[0]
    assert x.shape == (64,)

    packed = pack_documents([[1, 2, 3], [4, 5, 6]], 1024)
    assert len(packed) > 6

    train_loader, val_loader, tl, vl = create_dataloaders(data, 64, 4)
    x, y = next(iter(train_loader))
    assert x.shape == (4, 64)


if __name__ == "__main__":
    test("Tokenizer", test_tokenizer)
    test("Model Build", test_model_build)
    test("Forward Pass", test_forward_pass)
    test("Backward Pass", test_backward_pass)
    test("Optimizer", test_optimizer)
    test("Checkpoint Save/Load", test_checkpoint)
    test("Resume", test_resume)
    test("Generate", test_generate)
    test("LR Schedule", test_lr_schedule)
    test("Dataset", test_dataset)

    config = ModelConfig()
    model = NanoLLM(config)
    n_params = sum(p.numel() for p in model.parameters())

    print()
    print("=" * 50)
    print("  NanoLLM v2 Verification")
    print("=" * 50)
    for name, passed, error in results:
        status = "PASS" if passed else "FAIL"
        msg = f"  {status}  {name}"
        if error:
            msg += f" ({error})"
        print(msg)
    print()
    print(f"  Tests:        {PASS} passed, {FAIL} failed")
    print(f"  Parameters:   {n_params:,} ({n_params/1e6:.2f}M)")
    print(f"  Context:      {config.context_length}")
    print(f"  Vocabulary:   {config.vocab_size}")
    print("=" * 50)

    if FAIL > 0:
        sys.exit(1)
