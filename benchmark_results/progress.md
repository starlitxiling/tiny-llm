# Tiny LLM Optimization Benchmark Log

This file tracks local benchmark results while optimizing the `tiny_llm` implementation.

## Benchmark protocol

Unless noted otherwise, results use:

- Machine: local macOS Apple Silicon environment
- Device: GPU
- Model: `qwen3-0.6b` (`Qwen/Qwen3-0.6B-MLX-4bit`)
- Benchmark command: `pdm run bench`
- Request count: `--num-seqs 1`
- Prompt length: `--min-input-len 128 --max-input-len 128`
- Output length: `--min-output-len 65 --max-output-len 65`
- Warmup runs: `--warmup 2`
- Prefill logits: default runner setting, printed as `PrefillLogits=all`

## Results

| Date | Implementation | Loader | Checkpoint | Main optimization | Time (s) | Prompt tokens | Generated tokens | Output tok/s | Total tok/s | Prefill tok/s | Decode tok/s | Notes |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2026-08-13 | `tiny_llm` | `week1` | N/A | Full-prefix recomputation baseline | 105.60 | 128 | 65 | 0.62 | 1.83 | 148.14 | 0.61 | Recomputes the full prefix during decode. |
| 2026-08-13 | `tiny_llm` | `week2` | `kv-cache` | Per-layer KV cache + offset state flow | 2.38 | 128 | 65 | 27.37 | 81.25 | 1064.10 | 28.42 | Readable dense weights/RMSNorm/RoPE/SwiGLU/attention; only state flow changed. |

## Current speedup versus Week 1 baseline

Using the matched 128-token prompt / 65-token output benchmark above:

| Metric | Week 1 | Week 2 `kv-cache` | Speedup |
|---|---:|---:|---:|
| Output throughput | 0.62 tok/s | 27.37 tok/s | 44.15x |
| Total throughput | 1.83 tok/s | 81.25 tok/s | 44.40x |
| Prefill throughput | 148.14 tok/s | 1064.10 tok/s | 7.18x |
| Decode throughput | 0.61 tok/s | 28.42 tok/s | 46.59x |
| End-to-end time | 105.60 s | 2.38 s | 44.37x faster |

## Functional smoke test

Command:

```bash
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint kv-cache --model qwen3-0.6b
```

Observed output starts with a coherent response:

> A large language model (LLM) is a type of artificial intelligence (AI) system designed to understand and generate human-like text.

This is a smoke test only. Correctness should still be checked with the relevant `pdm run test --week ... --day ...` commands.

## Commands used for current entries

Week 2 KV-cache benchmark:

```bash
pdm run bench --solution tiny_llm --loader week2 \
  --week2-checkpoint kv-cache --model qwen3-0.6b \
  --num-seqs 1 --min-input-len 128 --max-input-len 128 \
  --min-output-len 65 --max-output-len 65 --warmup 2
```

Week 1 full-prefix baseline:

```bash
pdm run bench --solution tiny_llm --loader week1 \
  --model qwen3-0.6b \
  --num-seqs 1 --min-input-len 128 --max-input-len 128 \
  --min-output-len 65 --max-output-len 65 --warmup 2
```

## Future update template

Append one row to the Results table for each optimization checkpoint.

Recommended fields to record:

- Date
- Implementation / loader / checkpoint
- What changed since the previous row
- Exact benchmark command
- Time
- Prompt tokens
- Generated tokens
- Output throughput
- Total throughput
- Prefill throughput
- Decode throughput
- Any correctness tests run
- Notes about model, device, dependency, or implementation changes

Template row:

| Date | Implementation | Loader | Checkpoint | Main optimization | Time (s) | Prompt tokens | Generated tokens | Output tok/s | Total tok/s | Prefill tok/s | Decode tok/s | Notes |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| YYYY-MM-DD | `tiny_llm` | `week2` | `...` | ... | ... | 128 | 65 | ... | ... | ... | ... | ... |
