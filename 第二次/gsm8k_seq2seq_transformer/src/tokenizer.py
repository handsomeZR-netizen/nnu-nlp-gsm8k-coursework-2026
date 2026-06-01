import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


TOKEN_RE = re.compile(r"####|<<|>>|<nl>|[A-Za-z]+|\d|[+\-*/=.,!?;:()$%]|\S")
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<nl>"]


def tokenize(text: str, lowercase: bool = True) -> list[str]:
    if lowercase:
        text = text.lower()
    text = text.replace("\n", " <nl> ")
    return TOKEN_RE.findall(text)


@dataclass
class Vocab:
    stoi: dict[str, int]
    itos: list[str]
    lowercase: bool = True

    @property
    def pad_id(self) -> int:
        return self.stoi["<pad>"]

    @property
    def unk_id(self) -> int:
        return self.stoi["<unk>"]

    @property
    def bos_id(self) -> int:
        return self.stoi["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.stoi["<eos>"]

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(tok, self.unk_id) for tok in tokenize(text, self.lowercase)]

    def encode_tokens(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(tok, self.unk_id) for tok in tokens]

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        tokens = []
        for idx in ids:
            if idx < 0 or idx >= len(self.itos):
                tok = "<unk>"
            else:
                tok = self.itos[idx]
            if tok == "<eos>":
                break
            if skip_special and tok in {"<pad>", "<bos>"}:
                continue
            tokens.append(tok)
        return detokenize(tokens)

    def save(self, path: str | Path) -> None:
        payload = {"itos": self.itos, "lowercase": self.lowercase}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        itos = payload["itos"]
        return cls(stoi={tok: i for i, tok in enumerate(itos)}, itos=itos, lowercase=payload.get("lowercase", True))


def build_vocab(texts: list[str], min_freq: int = 1, lowercase: bool = True) -> Vocab:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text, lowercase))
    itos = SPECIAL_TOKENS[:]
    for tok, freq in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        if freq >= min_freq and tok not in itos:
            itos.append(tok)
    return Vocab(stoi={tok: i for i, tok in enumerate(itos)}, itos=itos, lowercase=lowercase)


def detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = text.replace(" <nl> ", "\n").replace("<nl>", "\n")
    text = re.sub(r"\s+([.,!?;:%)])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    text = text.replace("#### ", "#### ")
    text = text.replace(" << ", " <<").replace(" >> ", ">> ")
    text = text.replace(" / ", "/").replace(" * ", "*").replace(" + ", "+").replace(" - ", "-").replace(" = ", "=")
    text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)
    return re.sub(r"[ \t]+", " ", text).strip()
