# CPU inference comparison

Measured on AMD EPYC 9V74 80-Core Processor; Linux-6.17.0-1022-azure-x86_64-with-glibc2.39.
Python 3.12.15; PyTorch 2.14.1+cpu; FP32; 1 PyTorch thread.

Model: 2 layers, hidden width 128, 4 attention heads, MLP width 256, vocabulary 4,096.
Each case uses 3 warmups and 9 measured repetitions, with cached/uncached order alternating.
Both modes use only the final position's vocabulary projection. The full-prefix baseline
recomputes transformer states at every decode step; cached mode reuses prior keys/values.
16 output tokens means the first token is selected after prefill and 15 subsequent decode steps are timed.

| Batch | Prompt | Prefill ms | Cached decode ms | Uncached decode ms | Speedup | Batch decode tokens/s | Outputs match |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 32 | 0.597 | 4.882 | 10.793 | 2.21× | 3072 | Yes |
| 1 | 128 | 1.612 | 5.009 | 26.839 | 5.36× | 2995 | Yes |
| 2 | 32 | 0.908 | 6.356 | 16.473 | 2.59× | 4720 | Yes |
| 2 | 128 | 2.967 | 6.741 | 49.236 | 7.30× | 4450 | Yes |

All tested greedy outputs matched. Independent numerical tests also compare multi-step
cached logits with full-prefix logits for two input sequences.

These are CPU timings of an untrained small transformer, not GPU or production-model
serving results. Throughput aggregates batch tokens after the first output token.
Prefill timing excludes tokenization, transport, queueing and token selection.

[Raw samples, model costs and hardware metadata](cpu_inference_2026-10-15.json).
[Source workflow](https://github.com/At0508/transformer-inference-performance/actions/runs/38751897425), commit `ab56c36fa4ffdb411d7c1b9b247e74fb6ce00bb4`.
