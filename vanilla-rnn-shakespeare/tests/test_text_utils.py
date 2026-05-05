import sys
sys.path.insert(0, '.')
from utils.data_loader import load_text
from utils.text_utils import CharVocab

def test():
    text = load_text("vanilla-rnn-shakespeare/data/raw/shakespeare.txt")
    vocab = CharVocab(text)
    assert vocab.vocab_size == 65
    print("Text util test passed.")

if __name__ == "__main__":
    test()
