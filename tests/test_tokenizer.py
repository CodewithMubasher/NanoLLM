"""NanoLLM v2 - Tokenizer Tests"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer import train_tokenizer, load_tokenizer, encode, decode, tokenize_document, SPECIAL_TOKENS


def test_train_load(tmp_path=None):
    """Test tokenizer training and loading."""
    import tempfile
    if tmp_path is None:
        tmp_path = tempfile.mkdtemp()
    corpus = os.path.join(tmp_path, "corpus.txt")
    with open(corpus, "w", encoding="utf-8") as f:
        f.write("Hello world. How are you? This is a test. " * 100)
    path = os.path.join(tmp_path, "tokenizer.json")
    tok = train_tokenizer([corpus], path, vocab_size=1000)
    assert tok.get_vocab_size() <= 1000
    loaded = load_tokenizer(path)
    assert loaded is not None


def test_encode_decode_roundtrip():
    """Test encode/decode round-trip."""
    import tempfile
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
        tok = train_tokenizer([corpus], path, vocab_size=2000)

        tests = [
            "Hello world",
            "How are you",
            "Python is a programming language",
            "The sky is blue",
        ]
        for text in tests:
            ids = encode(text, tok, add_special=False)
            decoded = decode(ids, tok, skip_special=True)
            assert decoded.strip() == text.strip(), f"Round-trip failed: '{text}' -> '{decoded}'"


def test_special_tokens():
    """Test special tokens exist."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        corpus = os.path.join(tmp, "corpus.txt")
        with open(corpus, "w") as f:
            f.write("The quick brown fox. " * 100)
        path = os.path.join(tmp, "tok.json")
        tok = train_tokenizer([corpus], path, vocab_size=2000)
        for sp in SPECIAL_TOKENS:
            assert tok.token_to_id(sp) is not None, f"Missing special token: {sp}"


def test_no_bpe_markers():
    """Test that decoded output has no BPE markers."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        corpus = os.path.join(tmp, "corpus.txt")
        with open(corpus, "w") as f:
            f.write("The quick brown fox jumps over the lazy dog. ")
            f.write("Hello world, how are you doing today? " * 50)
        path = os.path.join(tmp, "tok.json")
        tok = train_tokenizer([corpus], path, vocab_size=2000)
        ids = encode("Hello world", tok, add_special=False)
        decoded = decode(ids, tok, skip_special=True)
        assert "Ġ" not in decoded, f"BPE marker found: {decoded}"
        assert "Ċ" not in decoded, f"BPE marker found: {decoded}"


def test_tokenize_document():
    """Test document tokenization adds BOS/EOS."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        corpus = os.path.join(tmp, "corpus.txt")
        with open(corpus, "w") as f:
            f.write("The quick brown fox. " * 100)
        path = os.path.join(tmp, "tok.json")
        tok = train_tokenizer([corpus], path, vocab_size=2000)
        ids = tokenize_document("Hello", tok)
        assert ids[0] == tok.token_to_id("<BOS>")
        assert ids[-1] == tok.token_to_id("<EOS>")


if __name__ == "__main__":
    test_train_load()
    test_encode_decode_roundtrip()
    test_special_tokens()
    test_no_bpe_markers()
    test_tokenize_document()
    print("All tokenizer tests passed!")
