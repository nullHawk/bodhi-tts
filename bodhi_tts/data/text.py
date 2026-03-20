import json
from typing import List, Dict


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


def build_vocab(texts: List[str], save_path: str) -> Dict[str, int]:
    """Scan all texts, collect unique chars, build and save vocabulary."""
    chars = set()
    for text in texts:
        chars.update(text.lower())
    sorted_chars = sorted(chars)

    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for i, ch in enumerate(sorted_chars):
        vocab[ch] = len(SPECIAL_TOKENS) + i

    with open(save_path, "w") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"Vocab built: {len(vocab)} tokens ({len(sorted_chars)} chars + {len(SPECIAL_TOKENS)} special)")
    return vocab


class CharTokenizer:
    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.id_to_char = {v: k for k, v in vocab.items()}

    @classmethod
    def from_vocab(cls, vocab_path: str) -> "CharTokenizer":
        with open(vocab_path) as f:
            vocab = json.load(f)
        return cls(vocab)

    def encode(self, text: str) -> List[int]:
        ids = [BOS_ID]
        for ch in text.lower():
            ids.append(self.vocab.get(ch, UNK_ID))
        ids.append(EOS_ID)
        return ids

    def decode(self, ids: List[int]) -> str:
        chars = []
        for i in ids:
            if i in (PAD_ID, BOS_ID, EOS_ID):
                continue
            chars.append(self.id_to_char.get(i, "?"))
        return "".join(chars)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
