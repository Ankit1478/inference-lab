# inference-lab

Hands-on LLM inference: predict speed and memory with maths, then measure it.

## Phase 1: Mac (Apple M5, 114 GB/s, 5.6 TFLOPS)

```bash
python3 -m venv venv && ./venv/bin/pip install mlx-lm
./venv/bin/python prefill.py                                                 # Llama 3.2 3B 4-bit
./venv/bin/python prefill.py --model mlx-community/Meta-Llama-3-8B-Instruct-4bit
./venv/bin/python decode.py                                                # 1 to 64 users
```

| File | Tests | Output |
|---|---|---|
| `prefill.py` | Prefill (TTFT) at 100–4,000 tokens, plus decode for 1 user | `prefill_<model>.csv` |
| `decode.py` | Decode with 1–64 users (batching, switch point) | `decode_log.txt` |

### Results (1,000-token prompt)

| | Llama 3.2 3B 4-bit | Llama 3 8B 4-bit |
|---|---|---|
| Model size | 1.81 GB | 5.27 GB |
| TTFT (maths / real) | 1.01 s / 0.79 s | 2.49 s / 2.11 s |
| Decode (maths / real) | 59 / 53 tok/s | 21 / 22 tok/s |
| Peak memory | 2.4 GB | 6.0 GB |

- Decode ≈ memory speed ÷ model size holds.
- TTFT is ~15–20% faster than maths: real compute is ~7 TFLOPS, not the 5.6 measured.
- Longer prompt → bigger KV cache → slower decode, more memory.

### Decode with many users (Llama 3.2 3B 4-bit)

| Users | Step time | Total tok/s | Per user tok/s | vs 1 user | Peak GB | Bound (maths) |
|---|---|---|---|---|---|---|
| 1 | 18 ms | 55 | 55.5 | 1.0× | 1.86 | memory |
| 4 | 29 ms | 141 | 35.1 | 2.5× | 2.04 | memory |
| 16 | 92 ms | 174 | 10.9 | 3.1× | 2.76 | compute |
| 64 | 152 ms | 421 | 6.6 | 7.6× | 5.42 | compute |

- More users → more total tokens/s (weights read once, shared), less per user.
- Switch point in users = 49 FLOPs/byte ÷ 3.6 (4-bit) ≈ 14 users.
- 4–16 users run well below the maths: MLX's small-batch 4-bit kernel doesn't share the weight read well. Full log: `decode_log.txt`.

### 4-bit vs 8-bit (Llama 3.2 3B, 1,000-token prompt)

| | 4-bit | 8-bit |
|---|---|---|
| Model size | 1.81 GB | 3.41 GB |
| TTFT | 0.80 s | 0.86 s |
| Decode (maths / real) | 59 / 52 tok/s | 32 / 32 tok/s |
| Memory speed used | 101 GB/s | 112 GB/s |
| Peak memory | 2.4 GB | 4.0 GB |

- Decode ≈ memory speed ÷ model size: 1.9× bigger model → 1.6× slower decode.
- TTFT barely changes: prefill is compute-bound, same maths in both.
- 4-bit gains less than 2×: it spends extra work turning 4-bit numbers back into 16-bit (dequantize).

## Phase 2: NVIDIA L4 GPU (Google Cloud, 300 GB/s, 121 TFLOPS 16-bit)

Llama 3 8B Instruct, 16-bit (16.06 GB), vLLM 0.33.

```bash
uv venv ~/venv --python 3.12 && source ~/venv/bin/activate && uv pip install vllm
export VLLM_USE_FLASHINFER_SAMPLER=0
python gpu_prefill.py      # TTFT at 100–4,000 tokens, plus decode for 1 user
python gpu_decode.py       # 1 to 128 users
```

| File | Tests | Output |
|---|---|---|
| `gpu_prefill.py` | Prefill (TTFT) + decode for 1 user | `gpu_prefill_<model>.csv` |
| `gpu_decode.py` | Decode with 1–128 users | `gpu_decode_log.txt` |

### Prefill (1 user)

| Prompt | TTFT maths | TTFT real | Decode maths | Decode real | Memory speed used |
|---|---|---|---|---|---|
| 100 | 0.012 s | 0.076 s | 18.7 | 16.9 tok/s | 271 GB/s |
| 1,000 | 0.115 s | 0.32 s | 18.5 | 16.8 tok/s | 271 GB/s |
| 4,000 | 0.461 s | 1.21 s | 18.1 | 16.3 tok/s | 271 GB/s |

- Decode hits ~90% of the maths: 271 of 300 GB/s used (MBU ≈ 90%).
- Prefill reaches only ~46 TFLOPS of 121 (MFU ≈ 40%). Likely the 72 W power limit lowering clocks, plus non-matmul work (attention, norms) the maths skips.
- Short prompts: even prefill must read all 16 GB once (≈ 53 ms), so TTFT for 100 tokens is memory-bound.

### Decode with many users

| Users | Step time (maths / real) | Total tok/s (maths / real) | Per user tok/s |
|---|---|---|---|
| 1 | 54 / 59 ms | 19 / 17 | 16.9 |
| 8 | 54 / 61 ms | 148 / 132 | 16.5 |
| 32 | 55 / 69 ms | 580 / 465 | 14.5 |
| 128 | 60 / 80 ms | 2,135 / 1,597 | 12.5 |

- 1 → 128 users: step time only 59 → 80 ms, total tok/s 17 → 1,597 (94×).
- Still memory-bound at 128: switch point = 403 FLOPs/byte ÷ 0.9 (16-bit) ≈ 432 users.

### Mac vs L4 (Llama 3 8B, 1,000-token prompt)

| | Mac M5 (4-bit, 5.3 GB) | L4 (16-bit, 16 GB) |
|---|---|---|
| TTFT | 2.11 s | 0.32 s (6.6× faster) |
| Decode, 1 user | 23.3 tok/s | 16.8 tok/s |
| Best total tok/s | 421 (3B, 64 users) | 1,597 (8B, 128 users) |

- Prefill: the GPU wins on compute.
- Decode for 1 user: the Mac wins because its 4-bit model is 3× smaller to read.
- Many users: the GPU wins easily; one weight read is shared by 128 users.

### Serving test: users arrive at random times (`serve_test.py`)

Model as a server (`vllm serve`), requests sent at random times like real users.
Each request: 500 input tokens, 100 output tokens.

```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct --max-model-len 4096 --no-enable-prefix-caching   # terminal 1
python serve_test.py                                                                             # terminal 2
```

**Maths (before running)**

| | Formula | Value |
|---|---|---|
| Max users at once | KV space ÷ tokens per user = 29,610 ÷ 600 | 49 |
| Prefill per request | 2 × 6.98B × 500 ÷ 46T (real compute) | 152 ms |
| Decode per request | 100 × 65 ms ÷ 49 users | 133 ms |
| Max requests/s | 1 ÷ (0.152 + 0.133) | 3.5 req/s |
| E2E when alone | 0.15 s + 100 × 54 ms | 5.5 s |

**Measured**

| Sent req/s | Done req/s | TTFT p50 / p99 | TPOT p50 / p99 | E2E p50 / p99 | Users in system |
|---|---|---|---|---|---|
| 1 | 0.7 | 0.29 / 0.39 s | 69 / 72 ms | 7.1 / 7.4 s | 5 |
| 2 | 1.7 | 0.37 / 0.72 s | 89 / 109 ms | 9.1 / 11.2 s | 15 |
| 3 | 2.4 | 0.59 / 1.92 s | 137 / 163 ms | 14.4 / 16.7 s | 33 |
| 4 | 2.7 | 2.22 / 4.67 s | 156 / 172 ms | 16.9 / 20.6 s | 44 |
| 6 | 2.9 | 8.72 / 26.59 s | 158 / 242 ms | 24.6 / 35.1 s | 74 |
| 8 | 3.0 | 21.59 / 43.57 s | 159 / 248 ms | 37.3 / 52.6 s | 102 |

- Max ~3.0 req/s (maths 3.5, ~85%). Sending more only grows the queue.
- TTFT blows up past ~4 req/s: 0.3 s → 21.6 s. Waiting explodes near 100% busy.
- Users in system (Little's Law = done req/s × avg E2E) reaches 102, but only ~49 fit in KV memory; the rest wait in line.
- TPOT 69 → 159 ms under load: new prompts' prefill runs inside the same steps as everyone's decode.
- "Done req/s" counts the time to finish the last requests, so it reads a bit low at light load.
- Safe operating point: ~2.5 req/s (~80% of max), TTFT stays under 1 s. Full log: `serve_log.txt`.
