import time, statistics, json, os
from openai import OpenAI
from prompt import PROMPTS

HOST  = os.getenv("HOST",  "http://localhost:8000")
MODEL = os.getenv("MODEL", "Qwen/Qwen3-0.6B")
LOG_FILE = os.getenv("LOG_FILE", "ttft_results.json")
NUM_REQUESTS = 50

client = OpenAI(base_url=f"{HOST}/v1", api_key="x")

def percentile(data, p):
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    return sorted_data[min(idx, len(sorted_data)-1)]

print(f"\nTTFT Benchmark — {MODEL}")
print("=" * 55)

ttfts = []
for i, prompt in enumerate(PROMPTS[:NUM_REQUESTS]):
    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=32,
        temperature=0.0,
        stream=True,
    )
    ttft = None
    for chunk in stream:
        if ttft is None and chunk.choices[0].delta.content:
            ttft = time.perf_counter() - t0
            break
    if ttft:
        ttfts.append(ttft)
        print(f"  {i+1:3d}  TTFT: {ttft:.3f}s  {prompt[:40]}")

p50  = percentile(ttfts, 50)
p75  = percentile(ttfts, 75)
p95  = percentile(ttfts, 95)
p99  = percentile(ttfts, 99)

print("\n" + "=" * 55)
print(f"  Requests    : {NUM_REQUESTS}")
print(f"  Avg TTFT    : {statistics.mean(ttfts):.3f}s")
print(f"  Median/p50  : {p50:.3f}s")
print(f"  p75         : {p75:.3f}s")
print(f"  p95         : {p95:.3f}s  <- production SLA metric")
print(f"  p99         : {p99:.3f}s")
print(f"  Min         : {min(ttfts):.3f}s")
print(f"  Max         : {max(ttfts):.3f}s")
print(f"  Std dev     : {statistics.stdev(ttfts):.3f}s")
print("=" * 55)

with open(LOG_FILE, "w") as f:
    json.dump({
        "avg": round(statistics.mean(ttfts),4),
        "p50": round(p50,4),
        "p75": round(p75,4),
        "p95": round(p95,4),
        "p99": round(p99,4),
        "min": round(min(ttfts),4),
        "max": round(max(ttfts),4),
        "std": round(statistics.stdev(ttfts),4),
        "all": [round(t,4) for t in ttfts]
    }, f, indent=2)
print(f"  Saved to {LOG_FILE}")
