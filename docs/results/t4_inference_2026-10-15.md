# Tesla T4 inference measurements — 15 October 2026

Measured in Google Colab using source commit `318a9e183f44b56cc5ca2468edbd8c31752c604e`, Python 3.13.15, PyTorch 2.11.0+cu130 and CUDA 13.0. GPU: Tesla T4, compute capability 7.5; NVIDIA driver 580.82.07. The model is an untrained, 855,040-parameter decoder with two layers, hidden size 128, four attention heads, FFN size 256 and vocabulary size 4096.

| Batch | Prompt tokens | Prefill median (ms) | Cached decode (ms/token) | Cached throughput (tokens/s) | Uncached/cached decode |
|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 1.478 | 1.159 | 862.9 | 1.298× |
| 1 | 128 | 1.447 | 1.059 | 944.6 | 1.317× |
| 2 | 32 | 1.465 | 1.081 | 1850.0 | 1.397× |
| 2 | 128 | 1.579 | 1.064 | 1879.8 | 1.364× |

All four cached/uncached greedy-output checks passed. The six existing unit tests also passed in this environment. Each case used three warmups and nine timed repetitions; the first generated token comes from prefill and the following 15 decode steps are measured separately. Both modes project only the final position to vocabulary logits. Raw samples, cache sizes and memory measurements are in [the JSON report](t4_inference_2026-10-15.json).

## Reproduce

Use the source commit above with a compatible CUDA PyTorch installation:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m inference_lab.benchmark --device cuda --dtype fp32 \
  --batch-sizes 1,2 --context-lengths 32,128 --output-tokens 16 \
  --warmup 3 --repetitions 9 --output results/t4_inference_fp32.json
```

## Interpretation

Caching improved decode by 1.30–1.40× in this run. This small model has substantial Python, framework and kernel-launch overhead, so reducing repeated arithmetic does not translate into the much larger speedups seen in the recorded CPU run. These measurements are not a GPU-versus-CPU comparison: those runs used different hosts and software stacks.

The batch-two throughput includes both sequences. Timings include the Python generation loop and cache concatenation. CUDA synchronization brackets measurements; no serving, tokenization or queueing costs are included. Clock frequencies were not locked on the shared Colab host. Some samples show outliers; all are retained. Peak allocated/reserved memory covers the complete case, including both modes, and cannot establish cached-mode memory savings. No language-quality or production-serving performance claim is made.
