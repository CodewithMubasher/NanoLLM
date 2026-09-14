"""
NanoLLM v2 - One-click training pipeline.
Usage: python run.py
"""
import os
import sys
import subprocess
import time
import json
import math
import torch


# ============================================================
# UI
# ============================================================

class C:
    R = "\033[0m"
    B = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GRN = "\033[32m"
    YLW = "\033[33m"
    CYN = "\033[36m"


def hdr(t):
    w = 50
    print(f"\n{C.CYN}{'─'*w}{C.R}")
    print(f"{C.CYN}{C.B}{t:^{w}}{C.R}")
    print(f"{C.CYN}{'─'*w}{C.R}")


def sec(t):
    print(f"\n{C.B}{t}{C.R}")
    print(f"{'─'*len(t)}")


def ok(msg):
    print(f"  {C.GRN}[✓]{C.R} {msg}")


def fail(msg):
    print(f"  {C.RED}[✗]{C.R} {msg}")


def info(msg):
    print(f"  {C.DIM}{msg}{C.R}")


def fmt_time(s):
    h, m, sec = int(s//3600), int((s%3600)//60), int(s%60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


# ============================================================
# Step 1: Install dependencies
# ============================================================

def install_deps():
    sec("Installing dependencies")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                          "torch", "tokenizers", "datasets", "numpy", "tqdm"])
    ok("Done")


# ============================================================
# Step 2: Verify GPU
# ============================================================

def verify_gpu():
    sec("GPU")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        ok(f"{name} ({vram:.1f} GB)")
        return "cuda"
    else:
        fail("No GPU - training will be slow")
        return "cpu"


# ============================================================
# Step 3: Preprocess datasets
# ============================================================

def preprocess():
    sec("Dataset")
    processed = "data/processed/pretraining.jsonl"

    if os.path.exists(processed):
        with open(processed) as f:
            n = sum(1 for _ in f)
        if n > 10000:
            ok(f"Already processed ({n:,} docs)")
            return

    from preprocess import preprocess_all
    preprocess_all()

    # Verify
    if not os.path.exists(processed):
        fail("Preprocessing failed - no output file")
        sys.exit(1)

    with open(processed) as f:
        n = sum(1 for _ in f)
    if n < 1000:
        fail(f"Too few documents ({n}) - check dataset download")
        sys.exit(1)

    ok(f"{n:,} documents ready")


# ============================================================
# Step 4: Train tokenizer on real data
# ============================================================

def train_tok():
    sec("Tokenizer")
    from tokenizer import train_tokenizer as tt, load_tokenizer

    os.makedirs("data/tokenizer", exist_ok=True)
    tok_path = "data/tokenizer/tokenizer.json"

    # Check if already trained with good vocab
    if os.path.exists(tok_path):
        tok = load_tokenizer(tok_path)
        v = tok.get_vocab_size()
        if v >= 10000:
            ok(f"Already trained (vocab: {v})")
            return
        else:
            info(f"Old vocab too small ({v}), retraining...")

    # Extract text from preprocessed data
    corpus_path = "data/tokenizer/corpus_for_tok.txt"
    info("Extracting text for tokenizer training...")

    count = 0
    with open("data/processed/pretraining.jsonl") as fin, \
         open(corpus_path, "w", encoding="utf-8") as fout:
        for line in fin:
            item = json.loads(line)
            text = item.get("text", "")
            if text and len(text) > 50:
                fout.write(text + "\n")
                count += 1
                if count >= 50000:
                    break

    info(f"Extracted {count:,} lines")

    # Train
    tok = tt([corpus_path], tok_path, vocab_size=16384)

    # Verify
    tok = load_tokenizer(tok_path)
    v = tok.get_vocab_size()
    if v < 5000:
        fail(f"Vocab too small ({v}) - corpus may be insufficient")
        sys.exit(1)

    ok(f"Vocab: {v}")


# ============================================================
# Step 5: Tokenize data for training
# ============================================================

def tokenize_data():
    sec("Tokenizing")
    tokenized_path = "data/processed/tokenized.jsonl"

    if os.path.exists(tokenized_path):
        with open(tokenized_path) as f:
            n = sum(1 for _ in f)
        if n > 10000:
            ok(f"Already tokenized ({n:,} docs)")
            return

    from tokenizer import load_tokenizer

    tok = load_tokenizer("data/tokenizer/tokenizer.json")
    vocab_size = tok.get_vocab_size()
    info(f"Vocab: {vocab_size}")

    count = 0
    total_tokens = 0
    rejected = 0

    with open("data/processed/pretraining.jsonl") as fin, \
         open(tokenized_path, "w", encoding="utf-8") as fout:
        for line in fin:
            item = json.loads(line)
            text = item.get("text", "")

            if not text or len(text) < 20:
                rejected += 1
                continue

            ids = tok.encode(text).ids

            # Validate
            if len(ids) < 5:
                rejected += 1
                continue

            fout.write(json.dumps({"ids": ids}) + "\n")
            count += 1
            total_tokens += len(ids)

    info(f"Tokenized {count:,} documents ({total_tokens:,} tokens)")
    info(f"Rejected {rejected:,} invalid documents")
    ok(f"Average {total_tokens//max(count,1)} tokens/doc")


# ============================================================
# Step 6: Preflight checks
# ============================================================

def preflight(device):
    sec("Pre-flight")
    from config import ModelConfig
    from model import NanoLLM
    from tokenizer import load_tokenizer

    checks = [
        ("Python", lambda: sys.version),
        ("PyTorch", lambda: torch.__version__),
        ("Tokenizer", lambda: load_tokenizer("data/tokenizer/tokenizer.json")),
        ("Dataset", lambda: os.path.exists("data/processed/tokenized.jsonl")),
    ]

    for name, fn in checks:
        try:
            fn()
            ok(name)
        except Exception as e:
            fail(f"{name}: {e}")
            sys.exit(1)

    try:
        config = ModelConfig()
        model = NanoLLM(config)
        n = sum(p.numel() for p in model.parameters())
        ok(f"Model: {n:,} params ({n/1e6:.2f}M)")
    except Exception as e:
        fail(f"Model: {e}")
        sys.exit(1)

    try:
        x = torch.randint(0, config.vocab_size, (2, 64))
        with torch.no_grad():
            logits, loss = model(x, x)
        ok(f"Forward: logits {list(logits.shape)}, loss {loss.item():.4f}")
    except Exception as e:
        fail(f"Forward: {e}")
        sys.exit(1)

    try:
        _, loss = model(x, x)
        loss.backward()
        ok("Backward")
    except Exception as e:
        fail(f"Backward: {e}")
        sys.exit(1)

    print(f"\n  {C.GRN}All checks passed.{C.R}")


# ============================================================
# Step 7: Train
# ============================================================

def train_model(device):
    from config import Config, effective_batch_size
    from model import NanoLLM, count_parameters
    from tokenizer import load_tokenizer
    from dataset import load_jsonl, pack_documents, create_dataloaders
    from train import save_checkpoint, get_lr
    from contextlib import nullcontext

    config = Config()
    config.train.device = device

    hdr("NanoLLM v2 Training")

    tokenizer = load_tokenizer(config.train.tokenizer_path)
    raw = load_jsonl("data/processed/tokenized.jsonl")
    token_ids = [item["ids"] for item in raw]
    packed = pack_documents(token_ids, config.model.context_length)
    train_loader, val_loader, train_n, val_n = create_dataloaders(
        packed, config.model.context_length, config.train.batch_size
    )

    model = NanoLLM(config.model).to(device)
    params = count_parameters(model)
    n_params = sum(v for k, v in params.items() if k != "Total")
    eff_bs = effective_batch_size(config)

    print(f"\n  {C.B}Model{C.R}")
    print(f"  {'─'*30}")
    print(f"  Parameters:   {n_params:,} ({n_params/1e6:.2f}M)")
    print(f"  Layers:       {config.model.num_layers}")
    print(f"  Hidden:       {config.model.hidden_size}")
    print(f"  Heads:        {config.model.num_heads}")
    print(f"  Context:      {config.model.context_length}")
    print(f"  Vocab:        {config.model.vocab_size}")

    print(f"\n  {C.B}Data{C.R}")
    print(f"  {'─'*30}")
    print(f"  Train tokens: {train_n/1e6:.1f}M")
    print(f"  Val tokens:   {val_n/1e6:.1f}M")

    print(f"\n  {C.B}Training{C.R}")
    print(f"  {'─'*30}")
    print(f"  Batch:        {config.train.batch_size} x {config.train.gradient_accumulation_steps} = {eff_bs}")
    print(f"  LR:           {config.train.learning_rate}")
    print(f"  Steps:        {config.train.max_steps}")
    print(f"  Warmup:       {config.train.warmup_steps}")

    use_amp = config.train.use_amp and device == "cuda"
    ctx = torch.amp.autocast(device_type=device, dtype=torch.float16) if use_amp else nullcontext()
    scaler = torch.amp.GradScaler(device) if use_amp else None

    optimizer = model.configure_optimizers(
        config.train.weight_decay, config.train.learning_rate,
        (config.train.beta1, config.train.beta2), device
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: get_lr(s, config.train.warmup_steps, config.train.max_steps,
                                   config.train.learning_rate, config.train.min_lr) / config.train.learning_rate,
    )

    # Resume
    step = 0
    best_val_loss = float("inf")
    resume_path = os.path.join(config.train.checkpoint_dir, "latest.pt")
    if os.path.exists(resume_path):
        from train import load_checkpoint
        step, best_val_loss = load_checkpoint(resume_path, model, optimizer, scaler, scheduler)
        print(f"\n  Resumed from step {step}")

    # Sanity test
    print(f"\n  {C.B}Sanity test{C.R}")
    model.train()
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    with ctx:
        _, loss = model(x, y)
    print(f"  Input:  {list(x.shape)}")
    print(f"  Loss:   {loss.item():.4f}")

    # Benchmark
    print(f"\n  {C.B}Benchmark{C.R}")
    model.train()
    start = time.time()
    for _ in range(20):
        with ctx:
            _, loss = model(x, y)
        loss.backward()
        model.zero_grad()
    elapsed = time.time() - start
    tok_per_sec = (x.numel() * 20) / elapsed
    steps_per_sec = 20 / elapsed
    remaining = (config.train.max_steps - step) / steps_per_sec
    print(f"  Speed:   {tok_per_sec/1000:.0f}K tok/s, {steps_per_sec:.1f} steps/s")
    print(f"  ETA:     {fmt_time(remaining)}")
    if device == "cuda":
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  VRAM:    {mem:.1f} GB")

    # Training loop
    print(f"\n  {C.GRN}Training...{C.R}\n")
    os.makedirs(config.train.checkpoint_dir, exist_ok=True)
    os.makedirs(config.train.logs_dir, exist_ok=True)

    log = []
    tokens_seen = 0
    t0 = time.time()
    running_loss = 0.0
    loss_steps = 0

    model.train()
    while step < config.train.max_steps:
        for x, y in train_loader:
            if step >= config.train.max_steps:
                break

            x, y = x.to(device), y.to(device)
            tokens_this = x.numel()

            with ctx:
                _, loss = model(x, y)
                loss = loss / config.train.gradient_accumulation_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item() * config.train.gradient_accumulation_steps
            loss_steps += 1

            if (step + 1) % config.train.gradient_accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            tokens_seen += tokens_this

            if step % config.train.log_interval == 0:
                avg = running_loss / max(loss_steps, 1)
                elapsed = time.time() - t0
                speed = tokens_seen / max(elapsed, 1)
                lr = scheduler.get_last_lr()[0]
                eta = (config.train.max_steps - step) / max(steps_per_sec, 0.01)
                bar_w = 25
                filled = int(bar_w * step / config.train.max_steps)
                bar = "█" * filled + "░" * (bar_w - filled)
                pct = 100 * step / config.train.max_steps

                vram = ""
                if device == "cuda":
                    mem = torch.cuda.memory_allocated() / 1e9
                    mem_t = torch.cuda.get_device_properties(0).total_memory / 1e9
                    vram = f"  {mem:.1f}/{mem_t:.1f}GB"

                print(
                    f"\r  Step {step:>6}/{config.train.max_steps}  {bar} {pct:>3.0f}%  "
                    f"Loss {avg:.4f}  LR {lr:.1e}  "
                    f"{speed/1000:.0f}K tok/s  ETA {fmt_time(eta)}{vram}  ",
                    end="", flush=True
                )
                running_loss = 0.0
                loss_steps = 0

            # Validation
            if step > 0 and step % config.train.eval_interval == 0:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx, vy = vx.to(device), vy.to(device)
                        with ctx:
                            _, vloss = model(vx, vy)
                        val_losses.append(vloss.item())
                model.train()
                vl = sum(val_losses) / len(val_losses)
                elapsed = time.time() - t0
                print(f"\n  {C.CYN}[Eval]{C.R} Step {step}  Val Loss {vl:.4f}  Time {fmt_time(elapsed)}")

                if vl < best_val_loss:
                    best_val_loss = vl
                    save_checkpoint(model, optimizer, scaler, scheduler, step, best_val_loss, config,
                                   os.path.join(config.train.checkpoint_dir, "best.pt"))
                    print(f"  {C.GRN}[Best]{C.R} Saved!")

                log.append({"step": step, "val_loss": vl})
                with open(os.path.join(config.train.logs_dir, "training_log.json"), "w") as f:
                    json.dump(log, f, indent=2)

            # Save
            if step > 0 and step % config.train.save_interval == 0:
                save_checkpoint(model, optimizer, scaler, scheduler, step, best_val_loss, config,
                               os.path.join(config.train.checkpoint_dir, "latest.pt"))

            step += 1

    save_checkpoint(model, optimizer, scaler, scheduler, step, best_val_loss, config,
                   os.path.join(config.train.checkpoint_dir, "latest.pt"))
    save_checkpoint(model, optimizer, scaler, scheduler, step, best_val_loss, config,
                   os.path.join(config.train.checkpoint_dir, "best.pt"))

    elapsed = time.time() - t0
    print(f"\n\n  {C.GRN}Training complete!{C.R}")
    print(f"  Steps:  {step}")
    print(f"  Time:   {fmt_time(elapsed)}")
    print(f"  Loss:   {best_val_loss:.4f}")

    return model


# ============================================================
# Step 8: Evaluate
# ============================================================

def evaluate(device):
    sec("Evaluation")
    from inference import load_model, generate
    from tokenizer import load_tokenizer

    ckpt = "checkpoints/best.pt"
    if not os.path.exists(ckpt):
        fail("No checkpoint found")
        return

    model, cfg, step, best_loss = load_model(ckpt, device)
    tokenizer = load_tokenizer("data/tokenizer/tokenizer.json")

    prompts = [
        "What is a computer?",
        "Why is the sky blue?",
        "Explain gravity simply.",
        "What is the capital of France?",
        "How do computers work?",
    ]

    results = []
    for p in prompts:
        r = generate(model, tokenizer, p, device, max_new_tokens=150, temperature=0.8, top_k=50)
        results.append({"prompt": p, "response": r})
        print(f"\n  Q: {p}")
        print(f"  A: {r[:200]}")

    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/samples.txt", "w") as f:
        f.write(f"NanoLLM v2 Evaluation\nStep: {step}\nBest loss: {best_loss:.4f}\n{'='*50}\n\n")
        for r in results:
            f.write(f"Q: {r['prompt']}\nA: {r['response']}\n{'─'*50}\n\n")

    ok("Saved to evaluation/samples.txt")


# ============================================================
# Main
# ============================================================

def main():
    hdr("NanoLLM v2")
    print(f"  {C.DIM}50M parameter language model{C.R}")

    install_deps()
    device = verify_gpu()
    preprocess()
    train_tok()
    tokenize_data()
    preflight(device)
    train_model(device)
    evaluate(device)

    hdr("Done")
    print(f"  Checkpoint: checkpoints/best.pt")
    print(f"  Inference:  python inference.py --checkpoint checkpoints/best.pt")
    print(f"  {C.DIM}GitHub: https://github.com/CodewithMubasher/NanoLLM{C.R}\n")


if __name__ == "__main__":
    main()
