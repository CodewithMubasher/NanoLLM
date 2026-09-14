"""
NanoLLM v2 - Configuration
Model and training configuration for ~50M parameter decoder-only language model.
"""
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    vocab_size: int = 16384
    hidden_size: int = 640
    num_layers: int = 8
    num_heads: int = 8
    head_dim: int = 80
    ffn_hidden_size: int = 1706
    context_length: int = 1024
    dropout: float = 0.0
    bias: bool = False
    tied_weights: bool = True

    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3


@dataclass
class TrainConfig:
    """Training configuration."""
    # Optimizer
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Schedule
    warmup_steps: int = 2000
    max_steps: int = 60000
    min_lr: float = 6e-5

    # Batch
    batch_size: int = 12
    gradient_accumulation_steps: int = 5

    # Mixed precision
    use_amp: bool = True

    # torch.compile
    compile_model: bool = True

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 1000
    eval_interval: int = 500
    log_interval: int = 10

    # Paths
    tokenizer_path: str = "data/tokenizer/tokenizer.json"
    processed_dir: str = "data/processed"
    logs_dir: str = "logs"
    eval_dir: str = "evaluation"

    # Resume
    resume_from: str = ""

    # Device
    device: str = ""


@dataclass
class Config:
    """Combined configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# Estimated effective batch size
def effective_batch_size(cfg: Config) -> int:
    return cfg.train.batch_size * cfg.train.gradient_accumulation_steps
