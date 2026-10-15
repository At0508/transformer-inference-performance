"""Transparent first-order compute and memory estimates for decoder inference."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelShape:
    layers: int = 2
    hidden_size: int = 128
    attention_heads: int = 4
    feed_forward_size: int = 256
    vocabulary_size: int = 4096
    max_sequence_length: int = 512
    kv_heads: int | None = None

    def __post_init__(self) -> None:
        if min(self.layers, self.hidden_size, self.attention_heads,
               self.feed_forward_size, self.vocabulary_size,
               self.max_sequence_length) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.hidden_size % self.attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        kv_heads = self.attention_heads if self.kv_heads is None else self.kv_heads
        if kv_heads <= 0 or self.attention_heads % kv_heads:
            raise ValueError("kv_heads must be a positive divisor of attention_heads")

    @property
    def effective_kv_heads(self) -> int:
        return self.attention_heads if self.kv_heads is None else self.kv_heads

    @property
    def head_dimension(self) -> int:
        return self.hidden_size // self.attention_heads

    @property
    def parameter_count(self) -> int:
        # Tied token embedding/output weights. LayerNorm affine parameters and
        # linear biases are included; optimizer state is not part of inference.
        d, f, v = self.hidden_size, self.feed_forward_size, self.vocabulary_size
        embedding = (v + self.max_sequence_length) * d
        kv_width = self.effective_kv_heads * self.head_dimension
        attention = 2 * d * d + 2 * d * kv_width + 2 * d + 2 * kv_width
        per_layer = attention + 2 * d * f + f + d + 4 * d
        final_norm = 2 * d
        return embedding + self.layers * per_layer + final_norm


@dataclass(frozen=True)
class Estimate:
    model: dict[str, int | None]
    batch_size: int
    prompt_tokens: int
    generated_tokens: int
    bytes_per_element: int
    parameter_count: int
    weight_bytes: int
    kv_cache_bytes_at_end: int
    prefill_flops: int
    prefill_logits_positions: int
    decode_flops: int
    decode_kv_read_bytes: int
    ideal_prefill_ms: float | None
    ideal_decode_ms: float | None


def estimate(
    shape: ModelShape,
    *,
    batch_size: int,
    prompt_tokens: int,
    generated_tokens: int,
    bytes_per_element: int = 2,
    prefill_logits_all_tokens: bool = True,
    peak_flops_per_second: float | None = None,
    memory_bandwidth_bytes_per_second: float | None = None,
) -> Estimate:
    """Estimate FLOPs and KV payload, with optional idealized scenario times.

    The model assumes dense projections, standard attention, tied input/output
    embeddings, and a KV cache. It excludes framework, allocator, padding,
    launch, host, and communication overheads. ``generated_tokens`` counts
    incremental decode forwards after prefill; prefill already supplies logits
    for the first output token. Zero decode forwards are supported.
    """
    if min(batch_size, prompt_tokens, bytes_per_element) <= 0 or generated_tokens < 0:
        raise ValueError("batch, prompt and bytes must be positive; decode steps cannot be negative")
    for name, value in (("peak_flops_per_second", peak_flops_per_second),
                        ("memory_bandwidth_bytes_per_second", memory_bandwidth_bytes_per_second)):
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive")

    d, f, h, kvh, layers = (shape.hidden_size, shape.feed_forward_size,
                            shape.attention_heads, shape.effective_kv_heads,
                            shape.layers)
    head_dim = shape.head_dimension
    kv_width = kvh * head_dim
    linear_flops_per_token_per_layer = 4 * d * d + 4 * d * kv_width + 4 * d * f
    output_projection_flops_per_token = 2 * d * shape.vocabulary_size
    prefill_logits_positions = prompt_tokens if prefill_logits_all_tokens else 1
    prefill_flops = batch_size * (
        layers * (prompt_tokens * linear_flops_per_token_per_layer
                  + 4 * prompt_tokens * prompt_tokens * d)
        + prefill_logits_positions * output_projection_flops_per_token
    )
    # Sum attention work as the cache grows token by token. This excludes the
    # small one-token causal-mask and sampling costs.
    decode_flops = batch_size * sum(
        layers * (linear_flops_per_token_per_layer
                  + 4 * (prompt_tokens + step + 1) * d)
        + output_projection_flops_per_token
        for step in range(generated_tokens)
    )
    parameter_count = shape.parameter_count
    weight_bytes = parameter_count * bytes_per_element
    kv_cache_bytes = (2 * layers * batch_size * (prompt_tokens + generated_tokens)
                      * kvh * head_dim * bytes_per_element)
    decode_kv_read_bytes = (2 * layers * batch_size * kvh * head_dim
                            * bytes_per_element * sum(
                                prompt_tokens + step for step in range(generated_tokens)
                            ))
    ideal_prefill_ms = None
    ideal_decode_ms = None
    if peak_flops_per_second:
        ideal_prefill_ms = 1000.0 * prefill_flops / peak_flops_per_second
        ideal_decode_ms = 1000.0 * decode_flops / peak_flops_per_second
    if memory_bandwidth_bytes_per_second:
        # A full-weight streaming scenario: prefill reads stored weights once,
        # while each decode step rereads them. Cache residency and sparse
        # position-table reads can reduce traffic, so this is not a guaranteed
        # lower bound on observed latency.
        prefill_memory_ms = 1000.0 * weight_bytes / memory_bandwidth_bytes_per_second
        decode_memory_ms = 1000.0 * (
            generated_tokens * weight_bytes + decode_kv_read_bytes
        ) / memory_bandwidth_bytes_per_second
        ideal_prefill_ms = max(ideal_prefill_ms or 0.0, prefill_memory_ms)
        ideal_decode_ms = max(ideal_decode_ms or 0.0, decode_memory_ms)

    return Estimate(
        model=asdict(shape), batch_size=batch_size,
        prompt_tokens=prompt_tokens, generated_tokens=generated_tokens,
        bytes_per_element=bytes_per_element, parameter_count=parameter_count,
        weight_bytes=weight_bytes, kv_cache_bytes_at_end=kv_cache_bytes,
        prefill_flops=prefill_flops, prefill_logits_positions=prefill_logits_positions,
        decode_flops=decode_flops,
        decode_kv_read_bytes=decode_kv_read_bytes,
        ideal_prefill_ms=ideal_prefill_ms, ideal_decode_ms=ideal_decode_ms,
    )
