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

`ttft_ms_model_only` is the measured prompt prefill time through the model's output logits. It excludes tokenization, request handling, and queueing. Decode throughput includes the Python loop, model calls, and on-device argmax selection, but excludes host-side token transfer. This small benchmark is useful for controlled comparisons, not production serving claims.

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

The optional roofline estimate is a lower bound under simplified assumptions. It omits launch overhead, cache residency, allocator padding, scheduling, and framework kernels. Compare it against GPU observations; do not describe the estimate itself as a measured latency.

## Validation and limits

The cache-equivalence check compares full causal prefill with prefill plus one incremental decode step. The cost-model checks validate KV scaling and input handling. The CUDA benchmark should be run on an NVIDIA GPU before making GPU performance claims. No production-model accuracy, model-serving throughput, or GPU timing is included in the source repository.

## License

MIT. See [LICENSE](LICENSE).
