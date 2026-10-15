"""Benchmark prefill and autoregressive decode, separating setup from timing."""

from __future__ import annotations

import argparse
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


def _measure(function, repetitions: int, device: torch.device) -> tuple[float, list[float]]:
    samples = []
    for _ in range(repetitions):
        _sync(device)
        start = time.perf_counter()
        function()
        _sync(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples), samples


def _benchmark_case(model: TinyDecoder, device: torch.device, dtype: torch.dtype,
                    batch: int, context: int, output_tokens: int,
                    warmup: int, repetitions: int, bytes_per_element: int) -> dict:
    tokens = torch.randint(0, model.embedding.num_embeddings, (batch, context),
                           device=device, dtype=torch.long)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    # Warm up with the same shapes, then synchronize before collecting samples.
    with torch.inference_mode():
        for _ in range(warmup):
            _, cache = model.prefill(tokens)
            next_token = tokens[:, -1]
            for _ in range(output_tokens):
                logits, cache = model.decode_step(next_token, cache)
                next_token = logits.argmax(dim=-1)
        _sync(device)

        prefill_ms, prefill_samples = _measure(
            lambda: model.prefill(tokens), repetitions, device
        )

        def decode_once() -> None:
            _, cache = model.prefill(tokens)
            next_token = tokens[:, -1]
            for _ in range(output_tokens):
                logits, cache = model.decode_step(next_token, cache)
                next_token = logits.argmax(dim=-1)

        decode_total_ms, decode_samples = _measure(decode_once, repetitions, device)

    shape = ModelShape(
        layers=len(model.blocks), hidden_size=model.blocks[0].hidden_size,
        attention_heads=model.blocks[0].heads,
        feed_forward_size=model.blocks[0].up.out_features,
        vocabulary_size=model.embedding.num_embeddings,
        max_sequence_length=model.max_sequence_length,
    )
    estimate_row = estimate(shape, batch_size=batch, prompt_tokens=context,
                            generated_tokens=output_tokens,
                            bytes_per_element=bytes_per_element)
    return {
        "batch_size": batch,
        "context_tokens": context,
        "generated_tokens": output_tokens,
        "prefill_ms_median": prefill_ms,
        "prefill_ms_samples": prefill_samples,
        "ttft_ms_model_only": prefill_ms,
        "decode_ms_per_token_median": decode_total_ms / output_tokens,
        "decode_ms_samples_total": decode_samples,
        "decode_tokens_per_second": 1000.0 * output_tokens / decode_total_ms,
        "kv_cache_bytes_at_end_estimate": estimate_row.kv_cache_bytes_at_end,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda" else None,
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
        if device.type == "cuda" else None,
        "note": "model-only timing; excludes tokenization, transport and request scheduling",
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
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--feed-forward-size", type=int, default=256)
    parser.add_argument("--vocabulary-size", type=int, default=4096)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--output", type=Path, default=Path("results/inference.json"))
    args = parser.parse_args()
    if args.output_tokens <= 0 or args.repetitions <= 0 or args.warmup < 0:
        parser.error("output tokens and repetitions must be positive; warmup cannot be negative")
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
    model = TinyDecoder(args.vocabulary_size, args.hidden_size, args.heads,
                        args.layers, args.max_sequence_length,
                        args.feed_forward_size).to(device=device, dtype=dtype)
    model.eval()
    rows = []
    for batch in args.batch_sizes:
        for context in args.context_lengths:
            rows.append(_benchmark_case(model, device, dtype, batch, context,
                                        args.output_tokens, args.warmup,
                                        args.repetitions, torch.empty((), dtype=dtype).element_size()))

    report = {
        "schema_version": 1,
        "model_kind": "randomly initialized small decoder; no language-quality claim",
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "cuda_runtime": torch.version.cuda},
        "hardware": {"device": str(device), "device_name": torch.cuda.get_device_name(device)
                     if device.type == "cuda" else platform.processor(),
                     "dtype": args.dtype},
        "benchmark": {"warmup": args.warmup, "repetitions": args.repetitions,
                      "timing_boundary": "prefill forward or autoregressive model decode only"},
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
