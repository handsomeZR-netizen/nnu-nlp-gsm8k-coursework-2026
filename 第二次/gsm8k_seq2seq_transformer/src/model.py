import random

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int, bidirectional: bool, dropout: float, pad_id: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.gru = nn.GRU(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

    @property
    def output_dim(self) -> int:
        return self.hidden_size * self.num_directions

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        outputs, hidden = self.gru(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, total_length=src.size(1))
        return outputs, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim: int, decoder_dim: int):
        super().__init__()
        self.enc_proj = nn.Linear(encoder_dim, decoder_dim, bias=False)
        self.dec_proj = nn.Linear(decoder_dim, decoder_dim, bias=False)
        self.v = nn.Linear(decoder_dim, 1, bias=False)

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor, src_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dec = self.dec_proj(decoder_hidden).unsqueeze(1)
        enc = self.enc_proj(encoder_outputs)
        scores = self.v(torch.tanh(enc + dec)).squeeze(-1)
        scores = scores.masked_fill(~src_mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        decoder_hidden: int,
        encoder_dim: int,
        num_layers: int,
        dropout: float,
        pad_id: int,
        use_attention: bool = True,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.use_attention = use_attention
        self.attention = BahdanauAttention(encoder_dim, decoder_hidden) if use_attention else None
        gru_input_dim = embedding_dim + (encoder_dim if use_attention else 0)
        self.gru = nn.GRU(
            gru_input_dim,
            decoder_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_input_dim = decoder_hidden + (encoder_dim if use_attention else 0) + embedding_dim
        self.out = nn.Linear(out_input_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward_step(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        embedded = self.dropout(self.embedding(input_token)).unsqueeze(1)
        context = None
        attn = None
        if self.use_attention and self.attention is not None:
            context, attn = self.attention(hidden[-1], encoder_outputs, src_mask)
            gru_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        else:
            gru_input = embedded
        output, hidden = self.gru(gru_input, hidden)
        output = output.squeeze(1)
        if context is not None:
            logits = self.out(torch.cat([output, context, embedded.squeeze(1)], dim=-1))
        else:
            logits = self.out(torch.cat([output, embedded.squeeze(1)], dim=-1))
        return logits, hidden, attn


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, config: dict):
        super().__init__()
        bidirectional = bool(config["model"]["bidirectional"])
        use_attention = config["model"].get("attention", "bahdanau") != "none"
        self.encoder = Encoder(
            vocab_size=vocab_size,
            embedding_dim=int(config["model"]["embedding_dim"]),
            hidden_size=int(config["model"]["encoder_hidden"]),
            num_layers=int(config["model"]["encoder_layers"]),
            bidirectional=bidirectional,
            dropout=float(config["model"]["dropout"]),
            pad_id=pad_id,
        )
        self.decoder_layers = int(config["model"]["decoder_layers"])
        self.decoder_hidden = int(config["model"]["decoder_hidden"])
        self.bridge = nn.Linear(self.encoder.output_dim, self.decoder_hidden)
        self.decoder = Decoder(
            vocab_size=vocab_size,
            embedding_dim=int(config["model"]["embedding_dim"]),
            decoder_hidden=self.decoder_hidden,
            encoder_dim=self.encoder.output_dim,
            num_layers=self.decoder_layers,
            dropout=float(config["model"]["dropout"]),
            pad_id=pad_id,
            use_attention=use_attention,
        )
        self.pad_id = pad_id

    def init_decoder_hidden(self, encoder_hidden: torch.Tensor) -> torch.Tensor:
        num_layers = self.encoder.num_layers
        num_dirs = self.encoder.num_directions
        batch = encoder_hidden.size(1)
        hidden = encoder_hidden.view(num_layers, num_dirs, batch, self.encoder.hidden_size).transpose(1, 2)
        hidden = hidden.reshape(num_layers, batch, self.encoder.output_dim)
        hidden = torch.tanh(self.bridge(hidden))
        if self.decoder_layers <= num_layers:
            hidden = hidden[-self.decoder_layers :]
        else:
            repeats = [hidden]
            while sum(x.size(0) for x in repeats) < self.decoder_layers:
                repeats.append(hidden[-1:].clone())
            hidden = torch.cat(repeats, dim=0)[-self.decoder_layers :]
        return hidden.contiguous()

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor, tgt: torch.Tensor, teacher_forcing_ratio: float) -> torch.Tensor:
        encoder_outputs, encoder_hidden = self.encoder(src, src_lens)
        hidden = self.init_decoder_hidden(encoder_hidden)
        src_mask = src.ne(self.pad_id)
        batch, tgt_len = tgt.size()
        logits = []
        input_token = tgt[:, 0]
        for step in range(1, tgt_len):
            step_logits, hidden, _ = self.decoder.forward_step(input_token, hidden, encoder_outputs, src_mask)
            logits.append(step_logits)
            use_teacher = random.random() < teacher_forcing_ratio
            input_token = tgt[:, step] if use_teacher else step_logits.argmax(dim=-1)
        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, src_lens: torch.Tensor, bos_id: int, eos_id: int, max_len: int) -> torch.Tensor:
        encoder_outputs, encoder_hidden = self.encoder(src, src_lens)
        hidden = self.init_decoder_hidden(encoder_hidden)
        src_mask = src.ne(self.pad_id)
        input_token = torch.full((src.size(0),), bos_id, dtype=torch.long, device=src.device)
        outputs = []
        finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            logits, hidden, _ = self.decoder.forward_step(input_token, hidden, encoder_outputs, src_mask)
            input_token = logits.argmax(dim=-1)
            outputs.append(input_token)
            finished |= input_token.eq(eos_id)
            if bool(finished.all()):
                break
        return torch.stack(outputs, dim=1) if outputs else torch.empty((src.size(0), 0), dtype=torch.long, device=src.device)

    @torch.no_grad()
    def beam_decode(
        self,
        src: torch.Tensor,
        src_lens: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int,
        beam_size: int,
        length_penalty: float = 0.7,
    ) -> torch.Tensor:
        if src.size(0) != 1:
            outputs = [
                self.beam_decode(src[i : i + 1], src_lens[i : i + 1], bos_id, eos_id, max_len, beam_size, length_penalty).squeeze(0)
                for i in range(src.size(0))
            ]
            max_out = max((item.numel() for item in outputs), default=0)
            padded = torch.full((len(outputs), max_out), eos_id, dtype=torch.long, device=src.device)
            for i, item in enumerate(outputs):
                padded[i, : item.numel()] = item
            return padded

        encoder_outputs, encoder_hidden = self.encoder(src, src_lens)
        init_hidden = self.init_decoder_hidden(encoder_hidden)
        src_mask = src.ne(self.pad_id)
        beams = [([], init_hidden, 0.0, False)]

        for _ in range(max_len):
            candidates = []
            all_finished = True
            for tokens, hidden, score, finished in beams:
                if finished:
                    candidates.append((tokens, hidden, score, True))
                    continue
                all_finished = False
                input_id = tokens[-1] if tokens else bos_id
                input_token = torch.tensor([input_id], dtype=torch.long, device=src.device)
                logits, next_hidden, _ = self.decoder.forward_step(input_token, hidden, encoder_outputs, src_mask)
                log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
                values, indices = torch.topk(log_probs, k=beam_size)
                for value, idx in zip(values.tolist(), indices.tolist()):
                    candidates.append((tokens + [idx], next_hidden.clone(), score + float(value), idx == eos_id))
            beams = sorted(candidates, key=lambda item: normalized_score(item[2], len(item[0]), length_penalty), reverse=True)[:beam_size]
            if all_finished:
                break

        best = max(beams, key=lambda item: normalized_score(item[2], len(item[0]), length_penalty))
        tokens = best[0]
        return torch.tensor(tokens, dtype=torch.long, device=src.device).unsqueeze(0)


def normalized_score(score: float, length: int, length_penalty: float) -> float:
    if length_penalty <= 0:
        return score
    return score / ((max(1, length) ** length_penalty))
