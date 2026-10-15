"""A small decoder-only transformer with explicit per-layer KV caching."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class DecoderBlock(nn.Module):
    def __init__(self, hidden_size: int, heads: int, feed_forward_size: int) -> None:
        super().__init__()
        if hidden_size % heads:
            raise ValueError("hidden_size must be divisible by heads")
        self.hidden_size = hidden_size
        self.heads = heads
        self.head_dim = hidden_size // heads
        self.norm_attention = nn.LayerNorm(hidden_size)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.norm_feed_forward = nn.LayerNorm(hidden_size)
        self.up = nn.Linear(hidden_size, feed_forward_size)
        self.down = nn.Linear(feed_forward_size, hidden_size)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query, key, value = self.qkv(self.norm_attention(x)).chunk(3, dim=-1)
        return self._split_heads(query), self._split_heads(key), self._split_heads(value)

    def _finish(self, x: torch.Tensor, query: torch.Tensor,
                key: torch.Tensor, value: torch.Tensor,
                past: tuple[torch.Tensor, torch.Tensor] | None,
                causal: bool) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if past is not None:
            key = torch.cat((past[0], key), dim=2)
            value = torch.cat((past[1], value), dim=2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if causal:
            query_length, key_length = scores.shape[-2:]
            q_positions = torch.arange(query_length, device=x.device) + key_length - query_length
            k_positions = torch.arange(key_length, device=x.device)
            mask = k_positions.unsqueeze(0) > q_positions.unsqueeze(1)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        probabilities = F.softmax(scores, dim=-1)
        context = torch.matmul(probabilities, value).transpose(1, 2).contiguous()
        context = context.view(x.shape[0], x.shape[1], self.hidden_size)
        x = x + self.output(context)
        x = x + self.down(F.gelu(self.up(self.norm_feed_forward(x))))
        return x, (key, value)

    def prefill(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        query, key, value = self._project(x)
        return self._finish(x, query, key, value, past=None, causal=True)

    def decode(self, x: torch.Tensor,
               past: tuple[torch.Tensor, torch.Tensor]
               ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        query, key, value = self._project(x)
        return self._finish(x, query, key, value, past=past, causal=False)


class TinyDecoder(nn.Module):
    """An untrained model for controlled inference mechanics experiments."""

    def __init__(self, vocabulary_size: int = 4096, hidden_size: int = 128,
                 heads: int = 4, layers: int = 2, max_sequence_length: int = 512,
                 feed_forward_size: int = 256) -> None:
        super().__init__()
        if min(vocabulary_size, hidden_size, heads, layers, max_sequence_length,
               feed_forward_size) <= 0:
            raise ValueError("model dimensions must be positive")
        self.embedding = nn.Embedding(vocabulary_size, hidden_size)
        self.position_embedding = nn.Embedding(max_sequence_length, hidden_size)
        self.max_sequence_length = max_sequence_length
        self.blocks = nn.ModuleList(
            DecoderBlock(hidden_size, heads, feed_forward_size) for _ in range(layers)
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, vocabulary_size, bias=False)
        self.output.weight = self.embedding.weight

    def prefill(self, token_ids: torch.Tensor
                ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        if token_ids.ndim != 2 or token_ids.shape[1] == 0:
            raise ValueError("token_ids must have shape [batch, nonzero sequence]")
        if token_ids.shape[1] > self.max_sequence_length:
            raise ValueError("prompt exceeds max_sequence_length")
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        x = self.embedding(token_ids) + self.position_embedding(positions).unsqueeze(0)
        cache = []
        for block in self.blocks:
            x, layer_cache = block.prefill(x)
            cache.append(layer_cache)
        return self.output(self.final_norm(x)), cache

    def decode_step(self, token_ids: torch.Tensor,
                    cache: list[tuple[torch.Tensor, torch.Tensor]]
                    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(1)
        if token_ids.ndim != 2 or token_ids.shape[1] != 1:
            raise ValueError("decode_step expects one token per batch row")
        if len(cache) != len(self.blocks):
            raise ValueError("cache must contain one key/value pair per layer")
        position = cache[0][0].shape[2]
        if position >= self.max_sequence_length:
            raise ValueError("decode would exceed max_sequence_length")
        positions = torch.full((token_ids.shape[0], 1), position,
                               device=token_ids.device, dtype=torch.long)
        x = self.embedding(token_ids) + self.position_embedding(positions)
        next_cache = []
        for block, layer_cache in zip(self.blocks, cache):
            x, new_layer_cache = block.decode(x, layer_cache)
            next_cache.append(new_layer_cache)
        logits = self.output(self.final_norm(x[:, -1]))
        return logits, next_cache
