"""
NanoLLM v2 - Evaluation
Run evaluation on fixed prompts and save results.
"""
import os
import json
import argparse
import torch

from config import ModelConfig
from model import NanoLLM
from tokenizer import load_tokenizer, BOS_ID
from inference import load_model, generate


EVAL_PROMPTS = [
    "Hello!",
    "How are you?",
    "What is a computer?",
    "Why is the sky blue?",
    "What is the capital of France?",
    "Explain gravity simply.",
    "What is the difference between a cat and a dog?",
    "Tell me something interesting.",
    "Why do people sleep?",
    "What is water?",
]


def evaluate(checkpoint_path: str, device: str = "", output_dir: str = "evaluation"):
    """Run evaluation on fixed prompts."""
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, config, step, best_val_loss = load_model(checkpoint_path, device)
    tokenizer = load_tokenizer("data/tokenizer/tokenizer.json")

    print(f"\nNanoLLM v2 Evaluation")
    print(f"{'='*50}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Step:       {step}")
    print(f"Best loss:  {best_val_loss:.4f}")
    print(f"Device:     {device}")
    print(f"{'='*50}\n")

    results = []
    for prompt in EVAL_PROMPTS:
        response = generate(model, tokenizer, prompt, device,
                          max_new_tokens=256, temperature=0.8, top_k=50, top_p=0.9)
        results.append({"prompt": prompt, "response": response})
        print(f"Prompt: {prompt}")
        print(f"Response: {response[:200]}...")
        print()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "samples.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"NanoLLM v2 Evaluation\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Step: {step}\n")
        f.write(f"Best val loss: {best_val_loss:.4f}\n")
        f.write(f"{'='*50}\n\n")
        for r in results:
            f.write(f"Prompt: {r['prompt']}\n")
            f.write(f"Response: {r['response']}\n")
            f.write(f"{'─'*50}\n\n")

    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"checkpoint": checkpoint_path, "step": step,
                   "best_val_loss": best_val_loss, "results": results}, f, indent=2)

    print(f"Results saved to {output_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NanoLLM v2 Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="evaluation")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.device, args.output_dir)
