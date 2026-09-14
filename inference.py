"""
NanoLLM v2 - Inference
Interactive text generation from a trained model checkpoint.
"""
import os
import sys
import argparse
import torch

from config import Config, ModelConfig
from model import NanoLLM
from tokenizer import load_tokenizer, BOS_ID


def load_model(checkpoint_path: str, device: str = ""):
    """Load model from checkpoint."""
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = ModelConfig(**ckpt["config"]["model"])
    model = NanoLLM(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    step = ckpt.get("step", 0)
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    return model, model_cfg, step, best_val_loss


@torch.no_grad()
def generate(model, tokenizer, prompt: str, device: str,
             max_new_tokens: int = 256, temperature: float = 0.8,
             top_k: int = 50, top_p: float = 0.9):
    """Generate text from a prompt."""
    tokens = tokenizer.encode(prompt).ids
    if tokens[0] != BOS_ID:
        tokens = [BOS_ID] + tokens
    x = torch.tensor([tokens], dtype=torch.long, device=device)

    output = model.generate(x, max_new_tokens=max_new_tokens,
                           temperature=temperature, top_k=top_k, top_p=top_p)
    output_ids = output[0].tolist()
    text = tokenizer.decode(output_ids, skip_special_tokens=True)
    return text


def interactive(model, tokenizer, device: str, config: ModelConfig):
    """Interactive chat loop."""
    print(f"\n{'='*50}")
    print(f"  NanoLLM v2")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Context:    {config.context_length}")
    print(f"  Device:     {device}")
    print(f"{'='*50}")
    print(f"  Commands: 'quit' to exit")
    print(f"            'temp <value>' to change temperature")
    print(f"            'topk <value>' to change top_k")
    print(f"            'topp <value>' to change top_p")
    print(f"            'max <value>' to change max_new_tokens")
    print(f"{'='*50}\n")

    temperature = 0.8
    top_k = 50
    top_p = 0.9
    max_new_tokens = 256

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt:
            continue

        if prompt.lower() == "quit":
            print("Goodbye!")
            break

        if prompt.lower().startswith("temp "):
            try:
                temperature = float(prompt.split()[1])
                print(f"  Temperature set to {temperature}")
            except (ValueError, IndexError):
                print("  Usage: temp 0.8")
            continue

        if prompt.lower().startswith("topk "):
            try:
                top_k = int(prompt.split()[1])
                print(f"  Top_k set to {top_k}")
            except (ValueError, IndexError):
                print("  Usage: topk 50")
            continue

        if prompt.lower().startswith("topp "):
            try:
                top_p = float(prompt.split()[1])
                print(f"  Top_p set to {top_p}")
            except (ValueError, IndexError):
                print("  Usage: topp 0.9")
            continue

        if prompt.lower().startswith("max "):
            try:
                max_new_tokens = int(prompt.split()[1])
                print(f"  Max new tokens set to {max_new_tokens}")
            except (ValueError, IndexError):
                print("  Usage: max 256")
            continue

        response = generate(model, tokenizer, prompt, device,
                          max_new_tokens=max_new_tokens,
                          temperature=temperature,
                          top_k=top_k,
                          top_p=top_p)
        print(f"Nano: {response}\n")


def main():
    parser = argparse.ArgumentParser(description="NanoLLM v2 Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    if not args.device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    model, config, step, best_val_loss = load_model(args.checkpoint, device)
    tokenizer = load_tokenizer("data/tokenizer/tokenizer.json")

    if args.prompt:
        response = generate(model, tokenizer, args.prompt, device,
                          max_new_tokens=args.max_new_tokens,
                          temperature=args.temperature,
                          top_k=args.top_k,
                          top_p=args.top_p)
        print(response)
    else:
        interactive(model, tokenizer, device, config)


if __name__ == "__main__":
    main()
