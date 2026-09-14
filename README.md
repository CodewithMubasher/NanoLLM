# NanoLLM v2

A ~50M parameter decoder-only language model trained from scratch on general English text.

## Architecture

```
Token Embedding
    ↓
[Transformer Block x 8]
    ├─ RMSNorm → Causal Self-Attention (RoPE) → Residual
    └─ RMSNorm → SwiGLU FFN → Residual
    ↓
RMSNorm → LM Head → Vocabulary Logits
```

### Components

| Component | Details |
|-----------|---------|
| Parameters | ~50M |
| Layers | 8 |
| Hidden Size | 576 |
| Attention Heads | 8 |
| Head Dimension | 72 |
| FFN Hidden | 1,536 |
| Context Length | 1,024 |
| Vocabulary | 16,384 |
| Normalization | RMSNorm |
| Position Encoding | RoPE |
| FFN Activation | SwiGLU |
| Weight Tying | Yes |

## Dataset

- **FineWeb-Edu** (~80%): Educational web text
- **DailyDialog** (~10%): Natural conversations
- **SQuAD** (~5%): Question-answer pairs
- **OpenBookQA** (~5%): Factual questions

## Installation

```bash
git clone https://github.com/CodewithMubasher/NanoLLM-v2.git
cd NanoLLM-v2
pip install -r requirements.txt
```

## Preprocessing

```bash
python preprocess.py
```

## Training

```bash
python train.py
```

### Resume from checkpoint

```bash
python train.py --resume checkpoints/latest.pt
```

### Configuration

Edit `config.py` to change model size, learning rate, batch size, etc.

## Evaluation

```bash
python evaluate.py --checkpoint checkpoints/best.pt
```

## Inference

```bash
python inference.py --checkpoint checkpoints/best.pt
```

## Google Colab

1. Open `colab_train.ipynb` in Colab
2. Run all cells in order
3. Training takes ~2 hours on Tesla T4

## Limitations

- 50M parameters is tiny compared to modern LLMs (GPT-3: 175B)
- Limited to 1,024 token context
- Will produce simple, sometimes incoherent text
- Not suitable for production use

## License

MIT
