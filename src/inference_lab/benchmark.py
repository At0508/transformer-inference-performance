"""Benchmark prefill and autoregressive decode, separating setup from timing."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from .cost_model import ModelShape, estimate
from .torch_model import TinyDecoder


def _csv_ints(value: str) -> list[int]:
    numbers = [int(part) for part in value.split(",") if part.strip()]
    if not numbers or min(numbers) <= 0:
        raise argparse.ArgumentTypeError("provide comma-separated positive integers")
    return numbers


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cpu_name() -> str:
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _measure(function, repetitions: int, device: torch.device) -> tuple[float, list[float]]:
    samples = []
    for _ in range(repetitions):
        _sync(device)
        start = time.perf_counter()
        function()
        _sync(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples), samples


def generate_cached(model: TinyDecoder, first_token: torch.Tensor, cache,
                    decode_steps: int) -> torch.Tensor:
    """Generate from an existing cache; no prompt forward is timed here."""
    generated = [first_token]
    next_token = first_token
    for _ in range(decode_steps):
        logits, cache = model.decode_step(next_token, cache)
        next_token = logits.argmax(dim=-1)
        generated.append(next_token)
    return torch.stack(generated, dim=1)


def generate_uncached(model: TinyDecoder, prompt: torch.Tensor,
                      first_token: torch.Tensor, decode_steps: int) -> torch.Tensor:
    """Reference baseline: recompute the entire prefix for every next token."""
    generated = [first_token]
    prefix = prompt
    next_token = first_token
    for _ in range(decode_steps):
        prefix = torch.cat((prefix, next_token.unsqueeze(1)), dim=1)
        logits, _ = model.prefill(prefix)
        next_token = logits[:, -1].argmax(dim=-1)
        generated.append(next_token)
    return torch.stack(generated, dim=1)


def _benchmark_case(model: TinyDecoder, device: torch.device,
                    batch: int, context: int, output_tokens: int,
                    warmup: int, repetitions: int, bytes_per_element: int) -> dict:
    tokens = torch.randint(0, model.embedding.num_embeddings, (batch, context),
                           device=device, dtype=torch.long)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    decode_steps = output_tokens - 1
    with torch.inference_mode():
        prompt_logits, base_cache = model.prefill(tokens)
        first_token = prompt_logits[:, -1].argmax(dim=-1)
        cached = lambda: generate_cached(model, first_token, base_cache, decode_steps)
        uncached = lambda: generate_uncached(model, tokens, first_token, decode_steps)
        if not torch.equal(cached(), uncached()):
            raise RuntimeError("cached and uncached greedy outputs disagree")
        actual_kv_bytes = sum(k.numel() * k.element_size() + v.numel() * v.element_size()
                              for k, v in base_cache)
        for _ in range(warmup):
            model.prefill(tokens)
            cached()
            uncached()
        _sync(device)

        prefill_ms, prefill_samples = _measure(
            lambda: model.prefill(tokens), repetitions, device
        )

        # Alternate measurement order to reduce systematic thermal/order bias.
        decode_samples, uncached_samples = [], []
        for index in range(repetitions):
            ordered = ((cached, decode_samples), (uncached, uncached_samples))
            for function, samples in ordered if index % 2 == 0 else reversed(ordered):
                _, values = _measure(function, 1, device)
                samples.extend(values)
        decode_total_ms = statistics.median(decode_samples)
        uncached_total_ms = statistics.median(uncached_samples)

    shape = ModelShape(
        layers=len(model.blocks), hidden_size=model.blocks[0].hidden_size,
        attention_heads=model.blocks[0].heads,
        feed_forward_size=model.blocks[0].up.out_features,
        vocabulary_size=model.embedding.num_embeddings,
        max_sequence_length=model.max_sequence_length,
    )
    estimate_row = estimate(shape, batch_size=batch, prompt_tokens=context,
                            generated_tokens=decode_steps,
                            bytes_per_element=bytes_per_element)
    return {
        "batch_size": batch,
        "context_tokens": context,
        "generated_tokens": output_tokens,
        "decode_steps_after_first_token": decode_steps,
        "greedy_outputs_match": True,
        "prefill_ms_median": prefill_ms,
        "prefill_ms_samples": prefill_samples,
        "ttft_ms_model_only": prefill_ms,
        "decode_ms_per_token_median": decode_total_ms / decode_steps,
        "decode_ms_samples_total": decode_samples,
        "decode_tokens_per_second": 1000.0 * batch * decode_steps / decode_total_ms,
        "uncached_decode_ms_samples_total": uncached_samples,
        "uncached_decode_ms_median": uncached_total_ms,
        "cached_decode_ms_median": decode_total_ms,
        "cached_vs_uncached_speedup": uncached_total_ms / decode_total_ms,
        "kv_cache_bytes_after_prefill_actual": actual_kv_bytes,
        "kv_cache_bytes_at_end_estimate": estimate_row.kv_cache_bytes_at_end,
        "analytical_cost": asdict(estimate_row),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda" else None,
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
        if device.type == "cuda" else None,
        "note": "decode excludes prefill; throughput aggregates batch tokens after the first output token",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--batch-sizes", type=_csv_ints, default=[1, 2])
    parser.add_argument("--context-lengths", type=_csv_ints, default=[32, 128])
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--feed-forward-size", type=int, default=256)
    parser.add_argument("--vocabulary-size", type=int, default=4096)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--output", type=Path, default=Path("results/inference.json"))
    args = parser.parse_args()
    if args.output_tokens < 2 or args.repetitions <= 0 or args.warmup < 0 or args.threads <= 0:
        parser.error("output tokens must be >=2; repetitions/threads positive; warmup nonnegative")
    if args.layers <= 0 or args.max_sequence_length <= 0:
        parser.error("layers and max sequence length must be positive")
    if max(args.context_lengths) + args.output_tokens > args.max_sequence_length:
        parser.error("max context length plus output tokens exceeds max sequence length")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device("cuda" if args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    dtype = {"fp32": torch.float32, "fp16": torch.float16,
             "bf16": torch.bfloat16}[args.dtype]
    if device.type == "cpu" and dtype != torch.float32:
        parser.error("CPU mode currently supports fp32 only; use CUDA for fp16/bf16 experiments")

    torch.manual_seed(7)
    torch.set_num_threads(args.threads)
    model = TinyDecoder(args.vocabulary_size, args.hidden_size, args.heads,
                        args.layers, args.max_sequence_length,
                        args.feed_forward_size).to(device=device, dtype=dtype)
    model.eval()
    rows = []
    for batch in args.batch_sizes:
        for context in args.context_lengths:
            rows.append(_benchmark_case(model, device, batch, context,
                                        args.output_tokens, args.warmup,
                                        args.repetitions, torch.empty((), dtype=dtype).element_size()))

    report = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_kind": "randomly initialized small decoder; no language-quality claim",
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "cuda_runtime": torch.version.cuda},
        "hardware": {"device": str(device), "device_name": torch.cuda.get_device_name(device)
                     if device.type == "cuda" else _cpu_name(),
                     "dtype": args.dtype, "torch_threads": torch.get_num_threads(),
                     "os": platform.platform()},
        "model": asdict(ModelShape(layers=args.layers, hidden_size=args.hidden_size,
                                  attention_heads=args.heads,
                                  feed_forward_size=args.feed_forward_size,
                                  vocabulary_size=args.vocabulary_size,
                                  max_sequence_length=args.max_sequence_length)),
        "benchmark": {"warmup": args.warmup, "repetitions": args.repetitions,
                      "timing_boundary": "prefill forward separately; decode excludes prompt forward",
                      "comparison": "cached decode vs full-prefix recomputation; alternating order",
                      "seed": 7},
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
