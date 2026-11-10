"""DECODE test on GPU (many users, vLLM): 1 to 128 users at once.
Same test as decode.py (Mac), now on an NVIDIA L4. For TTFT, see gpu_prefill.py.

KNOWN    L4 speeds, config, model size, KV per token, params
UNKNOWN  time per step, total tokens/s, tokens/s per user  -> we measure these

How we measure (vLLM batches all users together for us):
    t1        = time for all users to get 1 token        (prefill)
    t2        = time for all users to get 1+steps tokens (prefill + decode)
    step time = (t2 - t1) / steps

    python gpu_decode.py
    python gpu_decode.py --users 1,8,32
    python gpu_decode.py --fp8          (weights in 8-bit, embeddings stay 16-bit)
"""
import argparse, time
from transformers import AutoConfig
from vllm import LLM, SamplingParams

# NVIDIA L4
MEMORY_SPEED = 300e9     # bytes/s
COMPUTE_SPEED = 121e12   # FLOPS (16-bit, dense)
BYTES_PER_PARAM = 2      # 16-bit weights

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
parser.add_argument("--users", default="1,2,4,8,16,32,64,128")
parser.add_argument("--prompt", type=int, default=100, help="input tokens per user")
parser.add_argument("--steps", type=int, default=30, help="decode steps to time")
parser.add_argument("--fp8", action="store_true", help="8-bit weights (vLLM --quantization fp8)")
args = parser.parse_args()
LAYER_BYTES = 1 if args.fp8 else BYTES_PER_PARAM   # FP8 shrinks layer weights; embeddings + lm_head stay 16-bit

logfile = open("gpu_decode_log_fp8.txt" if args.fp8 else "gpu_decode_log.txt", "w")
def log(text=""):
    print(text)
    logfile.write(text + "\n")

# Config
cfg = AutoConfig.from_pretrained(args.model)
hidden = cfg.hidden_size
layers = cfg.num_hidden_layers
ffn = cfg.intermediate_size
vocab = cfg.vocab_size
kv_heads = cfg.num_key_value_heads
head_size = getattr(cfg, "head_dim", None) or hidden // cfg.num_attention_heads

# Formulas
params_per_layer = 2 * hidden * hidden + 2 * hidden * kv_heads * head_size + 3 * hidden * ffn
decode_params = layers * params_per_layer + hidden * vocab      # + lm_head (every new token needs it)
embed_params = hidden * vocab * (1 if cfg.tie_word_embeddings else 2)
model_size = layers * params_per_layer * LAYER_BYTES + embed_params * BYTES_PER_PARAM
kv_per_token = 2 * layers * kv_heads * head_size * 2
switch_point = COMPUTE_SPEED / MEMORY_SPEED                      # FLOPs per byte
flops_per_byte_per_user = 2 * decode_params / model_size        # 16-bit: ~1 FLOP per byte, FP8: ~2
switch_users = switch_point / flops_per_byte_per_user

log(f"\n{args.model}   {'FP8' if args.fp8 else '16-bit'}\n")
log("KNOWN")
log(f"  L4              {MEMORY_SPEED/1e9:.0f} GB/s memory, {COMPUTE_SPEED/1e12:.0f} TFLOPS compute")
log(f"  Model size      {model_size/1e9:.2f} GB  (read once per step, shared by all users)")
log(f"  Decode params   {decode_params/1e9:.2f}B  -> FLOPs per user per step = 2 x {decode_params/1e9:.2f}B = {2*decode_params/1e9:.1f} GFLOP")
log(f"  KV per token    {kv_per_token/1e6:.3f} MB  (each user has their own)")
log(f"  Switch point    {COMPUTE_SPEED/1e12:.0f}T / {MEMORY_SPEED/1e9:.0f}G = {switch_point:.0f} FLOPs per byte")
log(f"  Per user        2 x params / model size = {flops_per_byte_per_user:.1f} FLOPs per byte")
log(f"  Switch users    {switch_point:.0f} / {flops_per_byte_per_user:.1f} = ~{switch_users:.0f} users  <- after this, compute-bound")

# Model on GPU (prefix caching OFF, so every user does a real prefill)
llm = LLM(args.model, max_model_len=args.prompt + args.steps + 64, enable_prefix_caching=False,
          quantization="fp8" if args.fp8 else None)
prompt_tokens = llm.get_tokenizer().encode("The GPU reads the model weights from memory for every new word. " * 50)[:args.prompt]

def timed(users, n_tokens):
    params = SamplingParams(max_tokens=n_tokens, temperature=0, ignore_eos=True)
    start = time.perf_counter()
    llm.generate([{"prompt_token_ids": prompt_tokens}] * users, params, use_tqdm=False)  # all users in one batch
    return time.perf_counter() - start

def run(users):
    """Returns seconds per decode step (one step = one new token for EVERY user)."""
    t1 = timed(users, 1)                       # PREFILL all users
    t2 = timed(users, 1 + args.steps)          # PREFILL + DECODE steps
    return (t2 - t1) / args.steps

run(1)  # warm up

context = args.prompt + args.steps // 2                          # average tokens in each user's KV cache
summary = []
log("\nUNKNOWN (measured vs maths)")
for users in map(int, args.users.split(",")):
    step_time = run(users)

    bytes_per_step = model_size + users * kv_per_token * context
    memory_time = bytes_per_step / MEMORY_SPEED
    compute_time = 2 * decode_params * users / COMPUTE_SPEED
    maths_time = max(memory_time, compute_time)
    bound = "memory-bound" if memory_time > compute_time else "COMPUTE-BOUND"
    kv_total = users * kv_per_token * context

    log(f"\n--- {users} users ---")
    log(f"  Bytes read/step   = weights + users x KV per token x tokens")
    log(f"                    = {model_size/1e9:.2f} GB + {users} x {kv_per_token/1e6:.3f} MB x {context} = {bytes_per_step/1e9:.2f} GB")
    log(f"  Memory time       = bytes read / memory speed")
    log(f"                    = {bytes_per_step/1e9:.2f} GB / {MEMORY_SPEED/1e9:.0f} GB/s = {memory_time*1000:.1f} ms")

    log(f"  Compute time      = 2 x params x users / compute speed")
    log(f"                    = 2 x {decode_params/1e9:.2f}B x {users} / {COMPUTE_SPEED/1e12:.0f}T = {compute_time*1000:.1f} ms")
    log(f"  KV cache total    = {kv_total/1e9:.2f} GB")
    log(f"  Maths step time   = max(memory, compute) = {maths_time*1000:.1f} ms  ({bound})")
    log(f"  Measured step     = {step_time*1000:.1f} ms")
    log(f"  Total tokens/s    = {users / step_time:.0f}   (maths {users / maths_time:.0f})")
    log(f"  Tokens/s per user = {1 / step_time:.1f}   (maths {1 / maths_time:.1f})")
    summary.append((users, step_time, maths_time, bound))

log("\nSUMMARY")
log(f"  {'users':>5} | {'step ms':>8} {'maths':>7} | {'total tok/s':>11} {'maths':>7} | {'per user':>8} | bound")
for users, step_time, maths_time, bound in summary:
    log(f"  {users:>5} | {step_time*1000:>8.1f} {maths_time*1000:>7.1f} | {users/step_time:>11.0f} {users/maths_time:>7.0f} "
        f"| {1/step_time:>8.1f} | {bound}")
logfile.close()
print("\nSaved gpu_decode_log.txt")
