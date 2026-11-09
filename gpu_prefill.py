"""PREFILL test on GPU (1 user, vLLM): TTFT for different prompt lengths.
Same test as prefill.py (Mac), now on an NVIDIA L4. For many users, see gpu_decode.py.

KNOWN    L4 speeds, config (layers, hidden, KV heads, head size, FFN),
         model size, prefill params, KV per token
UNKNOWN  TTFT, decode tokens/s, memory speed used  -> we measure these

How we measure (vLLM gives no stopwatch inside):
    TTFT        = time to generate 1 token          (prefill + 1 step)
    decode tok/s = output / (time for 1+output tokens - TTFT)

    python gpu_prefill.py
    python gpu_prefill.py --model meta-llama/Llama-3.2-3B-Instruct
"""
import argparse, csv, time
from transformers import AutoConfig
from vllm import LLM, SamplingParams

# NVIDIA L4
MEMORY_SPEED = 300e9     # bytes/s
COMPUTE_SPEED = 121e12   # FLOPS (16-bit, dense)
BYTES_PER_PARAM = 2      # 16-bit weights

# Settings
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
parser.add_argument("--prompts", default="100,500,1000,2000,4000", help="input tokens")
parser.add_argument("--output", type=int, default=100, help="output tokens")
args = parser.parse_args()

# Config
cfg = AutoConfig.from_pretrained(args.model)
hidden = cfg.hidden_size
layers = cfg.num_hidden_layers
ffn = cfg.intermediate_size
vocab = cfg.vocab_size
kv_heads = cfg.num_key_value_heads
head_size = getattr(cfg, "head_dim", None) or hidden // cfg.num_attention_heads

# Formulas
params_per_layer = 2 * hidden * hidden + 2 * hidden * kv_heads * head_size + 3 * hidden * ffn  # QO + KV + FFN
prefill_params = layers * params_per_layer
embed_params = hidden * vocab * (1 if cfg.tie_word_embeddings else 2)                         # embeddings + lm_head
model_size = (prefill_params + embed_params) * BYTES_PER_PARAM
kv_per_token = 2 * layers * kv_heads * head_size * 2

print(f"\n{args.model}\n")
print("KNOWN")
print(f"  L4        {MEMORY_SPEED/1e9:.0f} GB/s memory, {COMPUTE_SPEED/1e12:.0f} TFLOPS compute")
print(f"  Config    {layers} layers, hidden {hidden}, {kv_heads} KV heads, head size {head_size}, FFN {ffn}")
print(f"  Formula   model {model_size/1e9:.2f} GB, prefill params {prefill_params/1e9:.2f}B, KV/token {kv_per_token/1e6:.3f} MB")

# Model on GPU (prefix caching OFF, so every run does a real prefill)
max_prompt = max(map(int, args.prompts.split(",")))
llm = LLM(args.model, max_model_len=max_prompt + args.output + 64, enable_prefix_caching=False)

# Prompt source: one long text, cut to any length
all_tokens = llm.get_tokenizer().encode("The GPU reads the model weights from memory for every new word. " * 2000)

def timed(prompt_len, n_tokens):
    params = SamplingParams(max_tokens=n_tokens, temperature=0, ignore_eos=True)
    start = time.perf_counter()
    llm.generate([{"prompt_token_ids": all_tokens[:prompt_len]}], params, use_tqdm=False)
    return time.perf_counter() - start

def run(prompt_len):
    ttft = timed(prompt_len, 1)                          # PREFILL: whole prompt read, first token out
    total = timed(prompt_len, 1 + args.output)           # PREFILL + DECODE
    decode = args.output / (total - ttft)                # DECODE: rest of the tokens, one by one
    return ttft, decode

run(100)  # warm up

print("\nUNKNOWN (measured vs maths)")
print(f"  {'prompt':>6} | {'TTFT':>7} {'maths':>7} | {'decode':>7} {'maths':>7} | {'GB/s used':>9}")
rows = []
for prompt_len in map(int, args.prompts.split(",")):
    ttft, decode = run(prompt_len)
    bytes_per_step = model_size + kv_per_token * prompt_len
    row = dict(
        prompt=prompt_len,
        # PREFILL (1 user)
        ttft=round(ttft, 3),
        ttft_maths=round(2 * prefill_params * prompt_len / COMPUTE_SPEED, 3),
        # DECODE (1 user)
        decode=round(decode, 1),
        decode_maths=round(MEMORY_SPEED / bytes_per_step, 1),
        gbps_used=round(bytes_per_step * decode / 1e9),
    )
    rows.append(row)
    print(f"  {row['prompt']:>6} | {row['ttft']:>6}s {row['ttft_maths']:>6}s | {row['decode']:>7} {row['decode_maths']:>7} "
          f"| {row['gbps_used']:>9}")

out_file = f"gpu_prefill_{args.model.split('/')[-1]}.csv"   # one file per model
with open(out_file, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0])
    w.writeheader()
    w.writerows(rows)
print(f"\nSaved {out_file}")
