"""
NanoLLM v2 - Training
Full training pipeline with pre-flight checks, clean UI, checkpointing, validation.
"""
import os
import sys
import time
import json
import math
import signal
import argparse
from contextlib import nullcontext

import torch
import numpy as np
from tqdm import tqdm

from config import Config, ModelConfig, TrainConfig, effective_batch_size
from model import NanoLLM, count_parameters
from tokenizer import load_tokenizer
from dataset import load_jsonl, pack_documents, create_dataloaders


# ============================================================
# Utilities
# ============================================================

class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"


def print_header(text: str):
    width = 50
    print(f"\n{Colors.CYAN}{'╭' + '─'*width + '╮'}{Colors.RESET}")
    print(f"{Colors.CYAN}│{Colors.RESET}{Colors.BOLD}{text:^{width}}{Colors.RESET}{Colors.CYAN}│{Colors.RESET}")
    print(f"{Colors.CYAN}{'╰' + '─'*width + '╯'}{Colors.RESET}\n")


def print_section(title: str):
    print(f"\n{Colors.BOLD}{title}{Colors.RESET}")
    print(f"{'─' * len(title)}")


def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_number(n: int) -> str:
    return f"{n:,}"


# ============================================================
# Learning Rate Schedule
# ============================================================

def get_lr(step: int, warmup_steps: int, max_steps: int, learning_rate: float, min_lr: float) -> float:
    if step < warmup_steps:
        return learning_rate * step / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(model, optimizer, scaler, scheduler, step, best_val_loss, config, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "step": step,
        "best_val_loss": best_val_loss,
        "config": {
            "model": config.model.__dict__,
            "train": config.train.__dict__,
        },
    }, path)


def load_checkpoint(path, model, optimizer=None, scaler=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt.get("step", 0), ckpt.get("best_val_loss", float("inf"))


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(model, val_loader, device, ctx):
    model.eval()
    losses = []
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float("inf")


# ============================================================
# Pre-flight Checks
# ============================================================

def run_preflight(config: Config):
    checks = []

    def check(name, fn):
        try:
            fn()
            checks.append((name, True, ""))
        except Exception as e:
            checks.append((name, False, str(e)))

    check("Python environment", lambda: sys.version)
    check("PyTorch", lambda: torch.__version__)

    def check_cuda():
        assert torch.cuda.is_available(), "CUDA not available"
    check("CUDA", check_cuda)

    def check_gpu():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"{name} ({vram:.1f} GB)"
    check("GPU", check_gpu)

    def check_tokenizer():
        assert os.path.exists(config.train.tokenizer_path), f"Tokenizer not found: {config.train.tokenizer_path}"
        load_tokenizer(config.train.tokenizer_path)
    check("Tokenizer", check_tokenizer)

    def check_dataset():
        path = os.path.join(config.train.processed_dir, "pretraining.jsonl")
        assert os.path.exists(path), f"Dataset not found: {path}"
    check("Dataset", check_dataset)

    def check_model():
        model = NanoLLM(config.model)
        return model
    check("Model construction", check_model)

    def check_forward():
        model = NanoLLM(config.model)
        x = torch.randint(0, config.model.vocab_size, (2, 64))
        ctx = nullcontext()
        with ctx:
            logits, loss = model(x, x)
        assert logits.shape == (2, 64, config.model.vocab_size)
    check("Forward pass", check_forward)

    def check_loss():
        model = NanoLLM(config.model)
        x = torch.randint(0, config.model.vocab_size, (2, 64))
        ctx = nullcontext()
        with ctx:
            _, loss = model(x, x)
        assert torch.isfinite(loss)
    check("Loss calculation", check_loss)

    def check_backward():
        model = NanoLLM(config.model)
        x = torch.randint(0, config.model.vocab_size, (2, 64))
        ctx = nullcontext()
        with ctx:
            _, loss = model(x, x)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert all(torch.isfinite(g).all() for g in grads)
    check("Backward pass", check_backward)

    def check_optimizer():
        model = NanoLLM(config.model)
        opt = model.configure_optimizers(
            config.train.weight_decay,
            config.train.learning_rate,
            (config.train.beta1, config.train.beta2),
            "cpu"
        )
    check("Optimizer", check_optimizer)

    def check_checkpoint_dir():
        os.makedirs(config.train.checkpoint_dir, exist_ok=True)
    check("Checkpoint directory", check_checkpoint_dir)

    print_section("Pre-flight Checks")
    all_passed = True
    for name, passed, error in checks:
        status = f"{Colors.GREEN}[✓]{Colors.RESET}" if passed else f"{Colors.RED}[✗]{Colors.RESET}"
        msg = f"  {status} {name}"
        if error:
            msg += f" - {Colors.RED}{error}{Colors.RESET}"
        print(msg)
        if not passed:
            all_passed = False

    if not all_passed:
        print(f"\n{Colors.RED}Pre-flight checks failed.{Colors.RESET}")
        sys.exit(1)

    print(f"\n{Colors.GREEN}Pre-flight checks passed.{Colors.RESET}")
    return True


# ============================================================
# Sanity Test
# ============================================================

def run_sanity_test(model, train_loader, device, ctx):
    print_section("Sanity Test")
    model.train()
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)

    print(f"  Input shape:   {list(x.shape)}")

    with ctx:
        logits, loss = model(x, y)

    print(f"  Logits shape:  {list(logits.shape)}")
    print(f"  Loss:          {loss.item():.4f}")
    assert torch.isfinite(loss), "Loss is not finite!"

    model.zero_grad()
    with ctx:
        _, loss = model(x, y)
    loss.backward()

    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.data.norm(2).item() ** 2
    grad_norm = grad_norm ** 0.5
    print(f"  Grad norm:     {grad_norm:.4f}")
    assert math.isfinite(grad_norm), "Gradient norm is not finite!"

    print(f"\n  {Colors.GREEN}Sanity test passed.{Colors.RESET}")


# ============================================================
# Benchmark
# ============================================================

def run_benchmark(model, train_loader, device, ctx, num_steps=20):
    print_section("Benchmark")
    model.train()
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)

    start = time.time()
    for _ in range(num_steps):
        with ctx:
            _, loss = model(x, y)
        loss.backward()
        model.zero_grad()
    elapsed = time.time() - start

    tokens_per_step = x.numel()
    tokens_per_sec = (tokens_per_step * num_steps) / elapsed
    steps_per_sec = num_steps / elapsed

    print(f"  Tokens/sec:    {tokens_per_sec/1000:.1f}K")
    print(f"  Steps/sec:     {steps_per_sec:.2f}")

    return steps_per_sec


# ============================================================
# Main Training
# ============================================================

def train(config: Config):
    os.makedirs(config.train.checkpoint_dir, exist_ok=True)
    os.makedirs(config.train.logs_dir, exist_ok=True)

    device = config.train.device
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print_header("NanoLLM v2 - 50M Parameter Language Model")

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Device:        {device}")
        print(f"  GPU:           {gpu_name}")
        print(f"  VRAM:          {vram:.1f} GB")
    else:
        print(f"  Device:        {device}")
        print(f"  {Colors.YELLOW}Warning: Training on CPU will be very slow{Colors.RESET}")

    run_preflight(config)

    tokenizer = load_tokenizer(config.train.tokenizer_path)
    data_path = os.path.join(config.train.processed_dir, "pretraining.jsonl")
    raw_data = load_jsonl(data_path)
    token_ids_list = [item["ids"] if "ids" in item else tokenizer.encode(item["text"]).ids for item in raw_data]

    packed = pack_documents(token_ids_list, config.model.context_length)
    train_loader, val_loader, train_tokens, val_tokens = create_dataloaders(
        packed, config.model.context_length, config.train.batch_size
    )

    model = NanoLLM(config.model).to(device)
    params = count_parameters(model)
    n_params = sum(v for k, v in params.items() if k != "Total")

    print_section("Model")
    print(f"  Parameters:    {format_number(n_params)}")
    print(f"  Layers:        {config.model.num_layers}")
    print(f"  Hidden:        {config.model.hidden_size}")
    print(f"  Heads:         {config.model.num_heads}")
    print(f"  Context:       {config.model.context_length}")
    print(f"  Vocabulary:    {config.model.vocab_size}")

    print_section("Dataset")
    print(f"  Train tokens:  {train_tokens/1e6:.1f}M")
    print(f"  Val tokens:    {val_tokens/1e6:.1f}M")

    eff_bs = effective_batch_size(config)
    print_section("Training")
    print(f"  Batch size:    {config.train.batch_size}")
    print(f"  Grad accum:    {config.train.gradient_accumulation_steps}")
    print(f"  Eff. batch:    {eff_bs}")
    print(f"  Learning rate: {config.train.learning_rate}")
    print(f"  Max steps:     {config.train.max_steps}")
    print(f"  Warmup steps:  {config.train.warmup_steps}")

    use_amp = config.train.use_amp and device == "cuda"
    ctx = torch.amp.autocast(device_type=device, dtype=torch.float16) if use_amp else nullcontext()
    scaler = torch.amp.GradScaler(device) if use_amp else None

    optimizer = model.configure_optimizers(
        config.train.weight_decay,
        config.train.learning_rate,
        (config.train.beta1, config.train.beta2),
        device,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(step, config.train.warmup_steps, config.train.max_steps, config.train.learning_rate, config.train.min_lr) / config.train.learning_rate,
    )

    step = 0
    best_val_loss = float("inf")

    if config.train.resume_from and os.path.exists(config.train.resume_from):
        step, best_val_loss = load_checkpoint(config.train.resume_from, model, optimizer, scaler, scheduler)
        print(f"\n  Resumed from:  {config.train.resume_from}")
        print(f"  Step:          {step}")
        print(f"  Best val loss: {best_val_loss:.4f}")

    steps_per_sec = run_benchmark(model, train_loader, device, ctx)
    total_steps = config.train.max_steps - step
    eta = total_steps / steps_per_sec if steps_per_sec > 0 else 0
    print(f"\n  Estimated time: {format_time(eta)}")

    run_sanity_test(model, train_loader, device, ctx)

    print(f"\n{Colors.GREEN}Starting training...{Colors.RESET}\n")

    training_log = []
    tokens_seen = 0
    start_time = time.time()
    running_loss = 0.0
    loss_steps = 0

    model.train()
    while step < config.train.max_steps:
        for x, y in train_loader:
            if step >= config.train.max_steps:
                break

            x, y = x.to(device), y.to(device)
            tokens_this_batch = x.numel()

            model.train()

            if config.train.gradient_accumulation_steps > 1:
                ctx_grad = torch.amp.autocast(device_type=device, dtype=torch.float16) if use_amp else nullcontext()
            else:
                ctx_grad = ctx

            with ctx_grad:
                _, loss = model(x, y)
                loss = loss / config.train.gradient_accumulation_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item() * config.train.gradient_accumulation_steps
            loss_steps += 1

            if (step + 1) % config.train.gradient_accumulation_steps == 0 or step == config.train.max_steps - 1:
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

            tokens_seen += tokens_this_batch

            if step % config.train.log_interval == 0:
                avg_loss = running_loss / max(loss_steps, 1)
                elapsed = time.time() - start_time
                tok_speed = tokens_seen / max(elapsed, 1)
                current_lr = scheduler.get_last_lr()[0]
                remaining = (config.train.max_steps - step) / max(steps_per_sec, 0.01)

                vram_used = ""
                if device == "cuda":
                    mem = torch.cuda.memory_allocated() / 1e9
                    mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                    vram_used = f"  VRAM {mem:.1f}/{mem_total:.1f} GB"

                progress = f"Step {step}/{config.train.max_steps}"
                bar_width = 30
                filled = int(bar_width * step / config.train.max_steps)
                bar = "█" * filled + "░" * (bar_width - filled)
                pct = 100 * step / config.train.max_steps

                print(
                    f"\r  {progress}  {bar} {pct:.0f}%  "
                    f"Loss {avg_loss:.4f}  LR {current_lr:.2e}  "
                    f"Tok/s {tok_speed/1000:.0f}K  ETA {format_time(remaining)}{vram_used}  ",
                    end="", flush=True
                )
                running_loss = 0.0
                loss_steps = 0

            if step > 0 and step % config.train.eval_interval == 0:
                val_loss = validate(model, val_loader, device, ctx)
                elapsed = time.time() - start_time
                print(f"\n  {Colors.CYAN}[Eval]{Colors.RESET} Step {step}  Val Loss: {val_loss:.4f}  Time: {format_time(elapsed)}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = os.path.join(config.train.checkpoint_dir, "best.pt")
                    save_checkpoint(model, optimizer, scaler, scheduler, step, best_val_loss, config, best_path)
                    print(f"  {Colors.GREEN}[Best]{Colors.RESET} New best model saved!")

                training_log.append({"step": step, "val_loss": val_loss, "time": elapsed})
                with open(os.path.join(config.train.logs_dir, "training_log.json"), "w") as f:
                    json.dump(training_log, f, indent=2)

            if step > 0 and step % config.train.save_interval == 0:
                latest_path = os.path.join(config.train.checkpoint_dir, "latest.pt")
                save_checkpoint(model, optimizer, scaler, scheduler, step, best_val_loss, config, latest_path)
                step_path = os.path.join(config.train.checkpoint_dir, f"step_{step:05d}.pt")
                save_checkpoint(model, optimizer, scaler, scheduler, step, best_val_loss, config, step_path)

            step += 1

    latest_path = os.path.join(config.train.checkpoint_dir, "latest.pt")
    save_checkpoint(model, optimizer, scaler, scheduler, step, best_val_loss, config, latest_path)

    elapsed = time.time() - start_time
    print(f"\n\n{Colors.GREEN}Training complete!{Colors.RESET}")
    print(f"  Steps:         {step}")
    print(f"  Time:          {format_time(elapsed)}")
    print(f"  Best val loss: {best_val_loss:.4f}")

    return model


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NanoLLM v2 Training")
    parser.add_argument("--resume", type=str, default="", help="Resume from checkpoint")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    config = Config()

    if args.resume:
        config.train.resume_from = args.resume
    if args.max_steps:
        config.train.max_steps = args.max_steps
    if args.batch_size:
        config.train.batch_size = args.batch_size
    if args.lr:
        config.train.learning_rate = args.lr
    if args.device:
        config.train.device = args.device

    train(config)
