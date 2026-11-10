"""SERVING test (vLLM server): users arrive at random times, like real life.
How many requests/s can one L4 handle before latency blows up?

KNOWN    L4 speeds, model size, KV per token, KV cache space (from vLLM log)
UNKNOWN  TTFT, TPOT, end-to-end time (p50 / p99), real requests/s  -> we measure these

Words
    TTFT   time to first token          (waiting in queue + prefill)
    TPOT   time per output token after the first (one decode step, shared with others)
    E2E    end-to-end: send -> last token
    p50    half of users were faster than this
    p99    99% of users were faster than this (the unlucky 1%)

Run (2 terminals on the VM)
    vllm serve meta-llama/Meta-Llama-3-8B-Instruct --max-model-len 4096 --no-enable-prefix-caching
    python serve_test.py
    python serve_test.py --rates 1,2,4
    FP8: add --quantization fp8 to vllm serve, then  python serve_test.py --fp8 --kv-tokens <from log>
    Prefix caching: drop --no-enable-prefix-caching, then  python serve_test.py --shared 400 ...
        (first 400 of the input tokens are the same "system prompt" for every request)
"""
import argparse, asyncio, random, statistics, time
from openai import AsyncOpenAI

# NVIDIA L4
MEMORY_SPEED = 300e9         # bytes/s
REAL_COMPUTE = 46e12         # FLOPS we actually got in gpu_prefill.py (not the 121 on paper)

# Llama 3 8B, 16-bit (from gpu_decode.py)
MODEL_SIZE = 16.06e9         # bytes  (FP8: 6.98B layer params x 1 byte + 1.05B embed/lm_head x 2 = 9.08 GB)
DECODE_PARAMS = 7.50e9
PREFILL_PARAMS = 6.98e9
KV_PER_TOKEN = 0.131e6       # bytes

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
parser.add_argument("--rates", default="1,2,3,4,6,8", help="requests per second to send")
parser.add_argument("--input", type=int, default=500, help="prompt tokens per request")
parser.add_argument("--output", type=int, default=100, help="output tokens per request")
parser.add_argument("--seconds", type=int, default=30, help="how long to send at each rate")
parser.add_argument("--kv-tokens", type=int, default=29610, help="'GPU KV cache size' line in vLLM log")
parser.add_argument("--fp8", action="store_true", help="server runs with --quantization fp8")
parser.add_argument("--shared", type=int, default=0, help="system prompt tokens, same for every request (cached)")
parser.add_argument("--max-seqs", type=int, default=256, help="vLLM max_num_seqs: most users it runs at once")
args = parser.parse_args()
if args.fp8:
    MODEL_SIZE = 9.08e9

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="none")

# ---------------- MATHS (before running) ----------------
new_tokens = args.input - args.shared                         # prompt tokens that still need prefill
tokens_per_user = new_tokens + args.output                    # shared prompt is stored once, not per user
max_users = min((args.kv_tokens - args.shared) // tokens_per_user, args.max_seqs)
avg_context = args.input + args.output // 2                   # decode still reads the whole context
step_time = (MODEL_SIZE + max_users * KV_PER_TOKEN * avg_context) / MEMORY_SPEED
prefill_time = 2 * PREFILL_PARAMS * new_tokens / REAL_COMPUTE
decode_share = args.output * step_time / max_users            # GPU time per request, decode part
max_rate = 1 / (prefill_time + decode_share)
e2e_alone = prefill_time + args.output * (MODEL_SIZE / MEMORY_SPEED)

print(f"\n{args.model}  {'FP8' if args.fp8 else '16-bit'}   input {args.input} ({args.shared} shared), output {args.output} tokens per request\n")
print("KNOWN -> MATHS")
print(f"  Max users at once  = min((KV space - shared) / (new input + output), max seqs)")
print(f"                     = min(({args.kv_tokens} - {args.shared}) / {tokens_per_user}, {args.max_seqs}) = {max_users} users")
print(f"  Step time (full)   = (weights + users x KV/token x tokens) / memory speed")
print(f"                     = ({MODEL_SIZE/1e9:.2f} GB + {max_users} x {KV_PER_TOKEN/1e6:.3f} MB x {avg_context}) / {MEMORY_SPEED/1e9:.0f} GB/s = {step_time*1000:.0f} ms")
print(f"  Prefill per request = 2 x prefill params x new input / real compute")
print(f"                     = 2 x {PREFILL_PARAMS/1e9:.2f}B x {new_tokens} / {REAL_COMPUTE/1e12:.0f}T = {prefill_time*1000:.0f} ms of GPU time")
print(f"  Decode per request = output x step time / users sharing it")
print(f"                     = {args.output} x {step_time*1000:.0f} ms / {max_users} = {decode_share*1000:.0f} ms of GPU time")
print(f"  Max requests/s     = 1 / (prefill + decode per request)")
print(f"                     = 1 / {(prefill_time + decode_share):.3f} s = {max_rate:.1f} req/s   <- latency blows up after this")
print(f"  E2E when alone     = prefill + output x (model size / memory speed)")
print(f"                     = {prefill_time:.2f} s + {args.output} x {MODEL_SIZE/MEMORY_SPEED*1000:.0f} ms = {e2e_alone:.1f} s")

# ---------------- MEASURE ----------------
SYSTEM_PROMPT = [random.Random(0).randint(1000, 30000) for _ in range(args.shared)]   # same for everyone

async def one_request():
    prompt = SYSTEM_PROMPT + [random.randint(1000, 30000) for _ in range(new_tokens)]   # own part: random, never cached
    start = time.perf_counter()
    first = None
    stream = await client.completions.create(
        model=args.model, prompt=prompt, max_tokens=args.output, temperature=0, stream=True,
        extra_body={"ignore_eos": True})
    async for _ in stream:
        if first is None:
            first = time.perf_counter()                 # first token arrived
    end = time.perf_counter()
    ttft = first - start
    tpot = (end - first) / (args.output - 1)
    return ttft, tpot, end - start

async def run(rate):
    """Send requests at random times (avg `rate` per second) for `seconds`. Wait for all."""
    tasks, start = [], time.perf_counter()
    while time.perf_counter() - start < args.seconds:
        tasks.append(asyncio.create_task(one_request()))
        await asyncio.sleep(random.expovariate(rate))   # random gap, average 1/rate
    results = await asyncio.gather(*tasks)
    return results, time.perf_counter() - start

def p(values, pct):
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * pct / 100))]

async def main():
    await asyncio.gather(*[one_request() for _ in range(2)])    # warm up

    print("\nUNKNOWN (measured)")
    print(f"  {'sent':>5} | {'done':>5} | {'out tok/s':>9} | {'TTFT p50':>8} {'p99':>6} | {'TPOT p50':>8} {'p99':>6} | {'E2E p50':>7} {'p99':>6} | {'users in':>8}")
    print(f"  {'req/s':>5} | {'req/s':>5} | {'':>9} | {'s':>8} {'s':>6} | {'ms':>8} {'ms':>6} | {'s':>7} {'s':>6} | {'system':>8}")
    for rate in map(float, args.rates.split(",")):
        results, total_time = await run(rate)
        ttft = [r[0] for r in results]
        tpot = [r[1] for r in results]
        e2e = [r[2] for r in results]
        done_rate = len(results) / total_time
        users_in = done_rate * statistics.mean(e2e)      # Little's Law: L = throughput x time in system
        print(f"  {rate:>5g} | {done_rate:>5.1f} | {done_rate*args.output:>9.0f} | "
              f"{p(ttft,50):>8.2f} {p(ttft,99):>6.2f} | {p(tpot,50)*1000:>8.0f} {p(tpot,99)*1000:>6.0f} | "
              f"{p(e2e,50):>7.1f} {p(e2e,99):>6.1f} | {users_in:>8.1f}")

    print("\n  users in system = done req/s x avg E2E   (Little's Law)")
    print(f"  Maths said: max ~{max_rate:.1f} req/s, ~{max_users} users at once, E2E alone ~{e2e_alone:.1f} s")

asyncio.run(main())
