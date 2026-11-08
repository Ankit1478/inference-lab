# inference-lab

Hands-on LLM inference: predict speed and memory with maths, then measure it.

## Phase 1: Mac (Apple M5, 114 GB/s, 5.6 TFLOPS)

```bash
python3 -m venv venv && ./venv/bin/pip install mlx-lm
./venv/bin/python bench.py                                                 # Llama 3.2 3B 4-bit
./venv/bin/python bench.py --model mlx-community/Meta-Llama-3-8B-Instruct-4bit
```

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
