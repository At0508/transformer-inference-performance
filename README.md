# Transformer Inference Performance

A compact decoder-only transformer used to study inference mechanics. The repository separates a transparent analytical model from a runnable PyTorch implementation with learned token/position embeddings and an explicit per-layer key/value (KV) cache. It measures prompt prefill and autoregressive decode independently across batch and context sizes.

The model is randomly initialized and intentionally small. It is a systems benchmark, not a language model, a quality evaluation, or a proxy for a production model. Its timing results apply only to the recorded machine and software stack.

## Questions this project answers

- How does prefill work differ from one-token-at-a-time decode?
- How do batch size, context length, precision, and cache size affect measured latency?
- How close are simple compute and memory bounds to measured model-only timings?
- Where do launch, framework, and Python overheads make the analytical model inaccurate?

## Setup

Python 3.10+ is required. Install the package and PyTorch for the environment you will use. Select a PyTorch wheel compatible with the installed driver and CUDA toolkit by following the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

```bash
python -m venv .venv
# Activate the environment, then install a compatible PyTorch build.
python -m pip install -e ".[torch]"
```

## Run inference measurements

```bash
python -m inference_lab.benchmark --device auto \
  --dtype fp32 --batch-sizes 1,2,4 --context-lengths 32,128 \
  --output-tokens 32 --warmup 3 --repetitions 10 \
  --output results/run.json
```

Use `--device cuda --dtype fp16` or `bf16` to run supported GPU precision experiments. The benchmark checks for a CUDA device and refuses an unavailable CUDA request. Use the same model dimensions, driver, PyTorch build, GPU, and run procedure when comparing cases. Run one variable sweep at a time and retain the generated JSON with the exact command.

`ttft_ms_model_only` is the measured prompt prefill time through the model's output logits. It excludes tokenization, request handling, and queueing. The first output token comes from those logits. `--output-tokens 16` then measures 15 incremental decode steps, excluding the prompt forward. Cached and uncached modes produce the same greedy output; the baseline recomputes the complete prefix at every step. Measurement order alternates between modes, and raw samples are retained.

`decode_tokens_per_second` aggregates tokens across the batch: `batch × decode_steps / elapsed_seconds`. It includes the Python loop, model calls, KV-cache concatenation, and argmax selection. This implementation copies cache tensors as they grow; it is not a preallocated production cache. CPU measurements use one PyTorch thread by default; change `--threads` explicitly and record the setting. This small benchmark is useful for controlled comparisons, not production serving claims.

Both modes project only the final position to vocabulary logits. The uncached baseline therefore repeats attention and decoder work, but does not waste time projecting logits for every prefix position. The model's default full-position logits remain available for numerical equivalence tests.

For a system trace on an NVIDIA machine:

```bash
nsys profile --trace=cuda,nvtx,osrt --output=results/inference \
  python -m inference_lab.benchmark --device cuda --context-lengths 32,128
```

## Analytical cost model

The model estimates dense projection and attention FLOPs, parameter and weight bytes, KV-cache payload, and cache reads during decode. With layers `L`, batch `B`, cached tokens `S`, KV heads `Hkv`, head dimension `D`, and bytes per element `b`, the cache payload is:

```text
KV bytes = 2 × L × B × S × Hkv × D × b
```

Run its checks with:

```bash
python -m unittest discover -s tests -v
```

The optional roofline calculation gives idealized scenario times using compute rate and a full-weight streaming assumption. Cache residency and sparse position-table reads can reduce actual traffic; framework/launch overhead can increase time. These estimates therefore need calibration and are not guaranteed latency bounds. The analytical API's `generated_tokens` counts decode forwards following prefill; the benchmark passes `output_tokens - 1`. The vocabulary projection is counted once after the decoder stack. Grouped-query attention dimensions are supported by the analytical model; the runnable PyTorch model uses standard multi-head attention. Compare estimates against observations; do not describe an estimate itself as measured latency.

CUDA peak allocated/reserved memory, when available, covers the entire case: correctness checks, warmups and both measurement modes. It is not an isolated measurement of cached-mode memory savings. KV-cache payload estimates are reported separately.

## Validation and limits

The checks compare cached and full-prefix logits across several decode steps and two sequences, verify tied parameter counts against PyTorch, and check batch throughput accounting. Cost-model checks validate hand-calculated FLOPs, KV scaling and input handling. GitHub Actions runs CPU PyTorch tests and publishes a CPU benchmark artifact. Run on an NVIDIA GPU before making GPU performance claims. No production-model accuracy or GPU timing is included in the source repository.

## Recorded CPU results

Recorded CPU measurements and all comparison cases are in the [inference results report](docs/results/cpu_inference_2026-10-15.md), with [raw timing samples and host metadata](docs/results/cpu_inference_2026-10-15.json). The benchmark compares KV caching with full-prefix recomputation using the same weights, input and final-token projection; every tested greedy output matched. These are CPU measurements of the small untrained model. GPU measurements remain to be collected.

## License

MIT. See [LICENSE](LICENSE).
