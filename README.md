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
| `prefill.py` | Prefill (TTFT) at 100–4,000 tokens, plus decode for 1 user | `prefill_results.csv` |
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
