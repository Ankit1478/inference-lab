"""PREFILL test (1 user): TTFT for different prompt lengths.
Also shows decode speed for 1 user. For many users, see decode.py.

KNOWN    Mac speeds, config (layers, hidden, KV heads, head size, FFN),
         model size, prefill params, KV per token
UNKNOWN  TTFT, decode tokens/s, memory speed used, peak memory  -> we measure these

OUTPUT (M5 Mac, Llama 3.2 3B 4-bit)
    KNOWN
      Mac       114 GB/s memory, 5.6 TFLOPS compute
      Config    28 layers, hidden 3072, 8 KV heads, head size 128, FFN 8192
      Formula   model 1.81 GB, prefill params 2.82B, KV/token 0.115 MB

    UNKNOWN (measured vs maths)
      prompt |    TTFT   maths |  decode   maths | GB/s used | peak GB
         100 |   0.15s    0.1s |    51.8    62.7 |        94 |    2.09
         500 |   0.46s    0.5s |    51.1    61.1 |        95 |    2.33
        1000 |   0.79s   1.01s |    52.6    59.3 |       101 |     2.4
        2000 |   1.64s   2.01s |    44.4    56.0 |        90 |    2.68
        4000 |   3.57s   4.03s |    41.5    50.3 |        94 |    2.89

    ./venv/bin/python prefill.py
    ./venv/bin/python prefill.py --model mlx-community/Meta-Llama-3-8B-Instruct-4bit
"""
import argparse, csv, time
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx.utils import tree_flatten

# My Mac
MEMORY_SPEED = 114e9    # bytes/s
COMPUTE_SPEED = 5.6e12  # FLOPS

# Settings
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct-4bit")
parser.add_argument("--prompts", default="100,500,1000,2000,4000", help="input tokens")
parser.add_argument("--output", type=int, default=100, help="output tokens")
args = parser.parse_args()

# Model + config
model, tokenizer = load(args.model)
cfg = model.args
hidden = cfg.hidden_size
layers = cfg.num_hidden_layers
ffn = cfg.intermediate_size
kv_heads = cfg.num_key_value_heads
head_size = getattr(cfg, "head_dim", None) or hidden // cfg.num_attention_heads

# Formulas
params_per_layer = 2 * hidden * hidden + 2 * hidden * kv_heads * head_size + 3 * hidden * ffn  # QO + KV + FFN
prefill_params = layers * params_per_layer
model_size = sum(w.nbytes for _, w in tree_flatten(model.parameters()))
kv_per_token = 2 * layers * kv_heads * head_size * 2

print(f"\n{args.model}\n")
print("KNOWN")
print(f"  Mac       {MEMORY_SPEED/1e9:.0f} GB/s memory, {COMPUTE_SPEED/1e12:.1f} TFLOPS compute")
print(f"  Config    {layers} layers, hidden {hidden}, {kv_heads} KV heads, head size {head_size}, FFN {ffn}")
print(f"  Formula   model {model_size/1e9:.2f} GB, prefill params {prefill_params/1e9:.2f}B, KV/token {kv_per_token/1e6:.3f} MB")

# Prompt source: one long text, cut to any length
all_tokens = tokenizer.encode("The GPU reads the model weights from memory for every new word. " * 2000)

def run(prompt_len):
    mx.reset_peak_memory()
    start, ttft = time.perf_counter(), None
    for out in stream_generate(model, tokenizer, all_tokens[:prompt_len], max_tokens=args.output):
        if ttft is None:                                # PREFILL done: whole prompt read, first token out
            ttft = time.perf_counter() - start
                                                        # DECODE: rest of the tokens, one by one
    return ttft, out

run(100)  # warm up

print("\nUNKNOWN (measured vs maths)")
print(f"  {'prompt':>6} | {'TTFT':>7} {'maths':>7} | {'decode':>7} {'maths':>7} | {'GB/s used':>9} | {'peak GB':>7}")
rows = []
for prompt_len in map(int, args.prompts.split(",")):
    ttft, out = run(prompt_len)
    bytes_per_step = model_size + kv_per_token * prompt_len
    row = dict(
        prompt=prompt_len,
        # PREFILL (1 user)
        ttft=round(ttft, 2),
        ttft_maths=round(2 * prefill_params * prompt_len / COMPUTE_SPEED, 2),
        # DECODE (1 user)
        decode=round(out.generation_tps, 1),
        decode_maths=round(MEMORY_SPEED / bytes_per_step, 1),
        gbps_used=round(bytes_per_step * out.generation_tps / 1e9),
        peak_gb=round(out.peak_memory, 2),
    )
    rows.append(row)
    print(f"  {row['prompt']:>6} | {row['ttft']:>6}s {row['ttft_maths']:>6}s | {row['decode']:>7} {row['decode_maths']:>7} "
          f"| {row['gbps_used']:>9} | {row['peak_gb']:>7}")

with open("prefill_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0])
    w.writeheader()
    w.writerows(rows)
print("\nSaved prefill_results.csv")
