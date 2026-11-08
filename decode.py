"""DECODE test (many users): 1 to 64 users at once.
Where does memory-bound turn into compute-bound? For TTFT, see prefill.py.

KNOWN    Mac speeds, config, model size, KV per token, params
UNKNOWN  time per step, total tokens/s, tokens/s per user, peak memory  -> we measure these

    ./venv/bin/python decode.py
    ./venv/bin/python decode.py --users 1,8,32
"""
import argparse, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx.utils import tree_flatten

# My Mac
MEMORY_SPEED = 114e9    # bytes/s
COMPUTE_SPEED = 5.6e12  # FLOPS

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct-4bit")
parser.add_argument("--users", default="1,2,4,8,16,32,64")
parser.add_argument("--prompt", type=int, default=100, help="input tokens per user")
parser.add_argument("--steps", type=int, default=30, help="decode steps to time")
args = parser.parse_args()

logfile = open("decode_log.txt", "w")
def log(text=""):
    print(text)
    logfile.write(text + "\n")

# Model + config
model, tokenizer = load(args.model)
cfg = model.args
hidden = cfg.hidden_size
layers = cfg.num_hidden_layers
ffn = cfg.intermediate_size
vocab = cfg.vocab_size
kv_heads = cfg.num_key_value_heads
head_size = getattr(cfg, "head_dim", None) or hidden // cfg.num_attention_heads

# Formulas
params_per_layer = 2 * hidden * hidden + 2 * hidden * kv_heads * head_size + 3 * hidden * ffn
decode_params = layers * params_per_layer + hidden * vocab      # + lm_head (every new token needs it)
model_size = sum(w.nbytes for _, w in tree_flatten(model.parameters()))
kv_per_token = 2 * layers * kv_heads * head_size * 2
switch_point = COMPUTE_SPEED / MEMORY_SPEED                      # FLOPs per byte
flops_per_byte_per_user = 2 * decode_params / model_size        # 4-bit: ~4 FLOPs per byte
switch_users = switch_point / flops_per_byte_per_user

log(f"\n{args.model}\n")
log("KNOWN")
log(f"  Mac             {MEMORY_SPEED/1e9:.0f} GB/s memory, {COMPUTE_SPEED/1e12:.1f} TFLOPS compute")
log(f"  Model size      {model_size/1e9:.2f} GB  (read once per step, shared by all users)")
log(f"  Decode params   {decode_params/1e9:.2f}B  -> FLOPs per user per step = 2 x {decode_params/1e9:.2f}B = {2*decode_params/1e9:.1f} GFLOP")
log(f"  KV per token    {kv_per_token/1e6:.3f} MB  (each user has their own)")
log(f"  Switch point    {COMPUTE_SPEED/1e12:.1f}T / {MEMORY_SPEED/1e9:.0f}G = {switch_point:.0f} FLOPs per byte")
log(f"  Per user        2 x params / model size = {flops_per_byte_per_user:.1f} FLOPs per byte (4-bit)")
log(f"  Switch users    {switch_point:.0f} / {flops_per_byte_per_user:.1f} = ~{switch_users:.0f} users  <- after this, compute-bound")

prompt_tokens = tokenizer.encode("The GPU reads the model weights from memory for every new word. " * 50)[:args.prompt]

def run(users):
    """Prefill all users together, then time decode steps. Returns seconds per step, peak GB."""
    cache = make_prompt_cache(model)
    x = mx.array([prompt_tokens] * users)                       # shape [users, prompt]
    logits = model(x, cache=cache)
    next_token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(next_token)
    mx.reset_peak_memory()

    for _ in range(3):                                          # warm up
        next_token = mx.argmax(model(next_token[:, None], cache=cache)[:, -1, :], axis=-1)
        mx.eval(next_token)

    start = time.perf_counter()
    for _ in range(args.steps):                                 # one step = one new token for EVERY user
        next_token = mx.argmax(model(next_token[:, None], cache=cache)[:, -1, :], axis=-1)
        mx.eval(next_token)
    step_time = (time.perf_counter() - start) / args.steps
    return step_time, mx.get_peak_memory() / 1e9

run(1)  # warm up

context = args.prompt + 3 + args.steps // 2                     # average tokens in each user's KV cache
summary = []
log("\nUNKNOWN (measured vs maths)")
for users in map(int, args.users.split(",")):
    step_time, peak_gb = run(users)

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
    log(f"                    = 2 x {decode_params/1e9:.2f}B x {users} / {COMPUTE_SPEED/1e12:.1f}T = {compute_time*1000:.1f} ms")
    log(f"  Maths step time   = bigger of memory time and compute time")
    log(f"                    = {maths_time*1000:.1f} ms  -> {bound}")
    log(f"  Real step time    = measured with stopwatch = {step_time*1000:.1f} ms")
    log(f"  Total tok/s       = users / step time")
    log(f"                    maths {users} / {maths_time:.4f} = {users/maths_time:.0f}  | real {users} / {step_time:.4f} = {users/step_time:.0f}")
    log(f"  Per user tok/s    = 1 / step time")
    log(f"                    maths 1 / {maths_time:.4f} = {1/maths_time:.1f}  | real 1 / {step_time:.4f} = {1/step_time:.1f}")
    log(f"  Real vs maths     = maths step time / real step time = {maths_time/step_time*100:.0f}%")
    log(f"  KV cache total    = users x KV per token x tokens")
    log(f"                    = {users} x {kv_per_token/1e6:.3f} MB x {context} = {kv_total/1e9:.3f} GB")
    log(f"  Peak memory       = measured = {peak_gb:.2f} GB")
    summary.append((users, step_time, maths_time, bound, peak_gb))

base = summary[0][1]
log("\nSUMMARY   (vs 1 user = total tok/s / total tok/s with 1 user)")
log(f"  {'users':>5} | {'step ms':>7} {'maths':>6} | {'total tok/s':>11} | {'per user':>8} | {'vs 1 user':>9} | {'peak GB':>7} | bound")
for users, step_time, maths_time, bound, peak_gb in summary:
    log(f"  {users:>5} | {step_time*1000:>7.1f} {maths_time*1000:>6.1f} | {users/step_time:>11.0f} | "
        f"{1/step_time:>8.1f} | {users*base/step_time:>8.1f}x | {peak_gb:>7.2f} | {bound}")
log("\nSaved decode_log.txt")
logfile.close()
