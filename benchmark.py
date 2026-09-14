"""
NanoLLM v2 - Benchmark
Quick training benchmark for time estimation.
"""
import os
import sys
import time
import argparse
import torch
from contextlib import nullcontext

from config import Config, ModelConfig, effective_batch_size
from model import NanoLLM
from tokenizer import load_tokenizer
from dataset import load_jsonl, pack_documents, create_dataloaders


def benchmark(config: Config, num_steps: int = 50):
    """Run a short training benchmark."""
    device = config.train.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nNanoLLM v2 Benchmark")
    print(f"{'='*50}")
    print(f"Device: {device}")

    tokenizer = load_tokenizer(config.train.tokenizer_path)
    data_path = os.path.join(config.train.processed_dir, "pretraining.jsonl")
    raw_data = load_jsonl(data_path)
    token_ids_list = [item["ids"] if "ids" in item else tokenizer.encode(item["text"]).ids for item in raw_data]

    packed = pack_documents(token_ids_list, config.model.context_length)
    train_loader, val_loader, train_tokens, val_tokens = create_dataloaders(
        packed, config.model.context_length, config.train.batch_size
    )

    model = NanoLLM(config.model).to(device)

    use_amp = config.train.use_amp and device == "cuda"
    ctx = torch.amp.autocast(device_type=device, dtype=torch.float16) if use_amp else nullcontext()
    scaler = torch.amp.GradScaler(device) if use_amp else None

    optimizer = model.configure_optimizers(
        config.train.weight_decay, config.train.learning_rate,
        (config.train.beta1, config.train.beta2), device
    )

    model.train()
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)

    print(f"  Batch shape: {list(x.shape)}")
    print(f"  Parameters:  {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Running {num_steps} steps...")

    start = time.time()
    for i in range(num_steps):
        with ctx:
            _, loss = model(x, y)
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if scaler:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    elapsed = time.time() - start
    tokens_per_step = x.numel()
    total_tokens = tokens_per_step * num_steps
    tokens_per_sec = total_tokens / elapsed
    steps_per_sec = num_steps / elapsed

    print(f"\nResults:")
    print(f"  Total time:   {elapsed:.1f}s")
    print(f"  Steps/sec:    {steps_per_sec:.2f}")
    print(f"  Tokens/sec:   {tokens_per_sec/1000:.1f}K")
    print(f"  Tokens/step:  {tokens_per_step:,}")

    if device == "cuda":
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak VRAM:    {mem:.1f} GB")

    effective_bs = effective_batch_size(config)
    total_steps = config.train.max_steps
    total_tokens_needed = total_tokens * (total_steps / num_steps)
    eta = total_steps / steps_per_sec

    print(f"\nEstimates for {total_steps} steps:")
    print(f"  Effective batch: {effective_bs}")
    print(f"  Total tokens:    {total_tokens_needed/1e9:.1f}B")
    print(f"  Estimated time:  {eta/3600:.1f} hours")

    return {
        "tokens_per_sec": tokens_per_sec,
        "steps_per_sec": steps_per_sec,
        "peak_vram_gb": mem if device == "cuda" else 0,
        "eta_hours": eta / 3600,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NanoLLM v2 Benchmark")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    config = Config()
    if args.device:
        config.train.device = args.device

    benchmark(config, args.steps)
