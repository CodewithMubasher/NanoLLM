"""
NanoLLM v2 - Tokenizer
Byte-level BPE tokenizer with proper encode/decode round-trip.
"""
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing


SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


def get_special_tokens():
    return SPECIAL_TOKENS


def train_tokenizer(corpus_files: list[str], save_path: str, vocab_size: int = 16384):
    """Train a BPE tokenizer on corpus files."""
    print(f"Training BPE tokenizer with vocab_size={vocab_size}...")

    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
        continuing_subword_prefix="",
    )

    tokenizer.train(corpus_files, trainer)

    tokenizer.decoder = ByteLevelDecoder()

    bos_id = tokenizer.token_to_id("<BOS>")
    eos_id = tokenizer.token_to_id("<EOS>")
    tokenizer.post_processor = TemplateProcessing(
        single=f"<BOS>:0 $A:0 <EOS>:0",
        pair=f"<BOS>:0 $A:0 <EOS>:0 $B:1 <EOS>:1",
        special_tokens=[("<BOS>", bos_id), ("<EOS>", eos_id)],
    )

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    return tokenizer


def load_tokenizer(tokenizer_path: str) -> Tokenizer:
    """Load a trained tokenizer from file."""
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    return Tokenizer.from_file(tokenizer_path)


def encode(text: str, tokenizer: Tokenizer, add_special: bool = True) -> list[int]:
    """Encode text to token IDs."""
    if add_special:
        return tokenizer.encode(text).ids
    else:
        encoding = tokenizer.encode(text)
        return encoding.ids


def decode(token_ids: list[int], tokenizer: Tokenizer, skip_special: bool = True) -> str:
    """Decode token IDs to text."""
    if skip_special:
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    else:
        return tokenizer.decode(token_ids, skip_special_tokens=False)


def tokenize_document(text: str, tokenizer: Tokenizer) -> list[int]:
    """Tokenize a document with BOS/EOS."""
    return tokenizer.encode(text).ids


def tokenize_documents(texts: list[str], tokenizer: Tokenizer) -> list[list[int]]:
    """Tokenize multiple documents."""
    return [tokenize_document(t, tokenizer) for t in texts]
