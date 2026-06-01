import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 1024):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, config: dict):
        super().__init__()
        model_cfg = config["model"]
        self.pad_id = pad_id
        self.d_model = int(model_cfg["d_model"])
        self.src_embedding = nn.Embedding(vocab_size, self.d_model, padding_idx=pad_id)
        self.tgt_embedding = nn.Embedding(vocab_size, self.d_model, padding_idx=pad_id)
        self.positional = PositionalEncoding(
            d_model=self.d_model,
            dropout=float(model_cfg["dropout"]),
            max_len=int(model_cfg.get("max_position", 1024)),
        )
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=int(model_cfg["nhead"]),
            num_encoder_layers=int(model_cfg["encoder_layers"]),
            num_decoder_layers=int(model_cfg["decoder_layers"]),
            dim_feedforward=int(model_cfg["dim_feedforward"]),
            dropout=float(model_cfg["dropout"]),
            batch_first=True,
            norm_first=True,
        )
        self.output = nn.Linear(self.d_model, vocab_size)

    def _embed_src(self, src: torch.Tensor) -> torch.Tensor:
        return self.positional(self.src_embedding(src) * math.sqrt(self.d_model))

    def _embed_tgt(self, tgt: torch.Tensor) -> torch.Tensor:
        return self.positional(self.tgt_embedding(tgt) * math.sqrt(self.d_model))

    @staticmethod
    def _causal_mask(size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src_padding_mask = src.eq(self.pad_id)
        memory = self.transformer.encoder(self._embed_src(src), src_key_padding_mask=src_padding_mask)
        return memory, src_padding_mask

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, src_padding_mask: torch.Tensor) -> torch.Tensor:
        tgt_padding_mask = tgt.eq(self.pad_id)
        tgt_mask = self._causal_mask(tgt.size(1), tgt.device)
        decoded = self.transformer.decoder(
            self._embed_tgt(tgt),
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        return self.output(decoded)

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        del src_lens
        memory, src_padding_mask = self.encode(src)
        decoder_input = tgt[:, :-1].contiguous()
        return self.decode(decoder_input, memory, src_padding_mask)

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, src_lens: torch.Tensor, bos_id: int, eos_id: int, max_len: int) -> torch.Tensor:
        del src_lens
        memory, src_padding_mask = self.encode(src)
        generated = torch.full((src.size(0), 1), bos_id, dtype=torch.long, device=src.device)
        finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            logits = self.decode(generated, memory, src_padding_mask)[:, -1, :]
            next_token = logits.argmax(dim=-1)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            finished |= next_token.eq(eos_id)
            if bool(finished.all()):
                break
        return generated[:, 1:]
