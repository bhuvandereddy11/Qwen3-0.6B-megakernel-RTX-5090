from prompt import PROMPTS
import os
import json
import statistics
from pathlib import Path
import time
from openai import OpenAI

def load_env(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

load_env()

HOST       = os.getenv("HOST", "http://localhost:8000")
MODEL      = os.getenv("MODEL", "Qwen/Qwen3-0.6B")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "128"))
ISL_TARGET = int(os.getenv("ISL_TARGET", "100"))
LOG_FILE   = os.getenv("LOG_FILE", "isl_benchmark.json")

client = OpenAI(base_url=f"{HOST}/v1", api_key="vllm-key")

PADDING = (
    "The following is a general knowledge question. "
    "Please read the context carefully before answering. "
    "This is a test of knowledge retrieval and factual accuracy. "
    "The context provided is for padding purposes to simulate longer input sequences. "
    "Machine learning models process tokens sequentially and longer inputs test prefill performance. "
    "Neural networks use attention mechanisms to relate tokens to each other across the sequence. "
    "Large language models like Qwen are trained on diverse datasets covering many topics. "
    "Inference speed depends on both input length and output length in transformer architectures. "
    "The key value cache stores intermediate computations to speed up autoregressive generation. "
    "Now please answer the following question accurately and concisely: "
)

ISL_PADDING = {
    20:   "",
    100:  PADDING[:200],
    500:  PADDING * 5,
    1000: PADDING * 10,
}

def build_prompt(question: str, isl_target: int) -> str:
    pad = ISL_PADDING.get(isl_target, "")
    return f"{pad}{question}"

def server_ready():
    import urllib.request
    print(f"[isl_bench] waiting for server at {HOST}", end="", flush=True)
    for _ in range(60):
        try:
            with urllib.request.urlopen(f"{HOST}/health", timeout=3):
                print("ready\n")
                return
        except Exception:
            print(".", end="", flush=True)
            time.sleep(3)
    raise RuntimeError("Server not ready")

def run_isl_benchmark():
    print("=" * 65)
    print("ISL Benchmark — Input Sequence Length Test")
    print("=" * 65)
    print(f"  Server     : {HOST}")
    print(f"  Model      : {MODEL}")
    print(f"  ISL Target : ~{ISL_TARGET} tokens")
    print(f"  Max Output : {MAX_TOKENS} tokens")
    print(f"  Requests   : {len(PROMPTS)}")
    print("=" * 65 + "\n")

    server_ready()

    latencies = []
    in_tokens_list = []
    out_tokens_list = []
    results = []

    for i, question in enumerate(PROMPTS):
        prompt = build_prompt(question, ISL_TARGET)

        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            stream=False,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
        latency = time.perf_counter() - t0

        in_tok  = getattr(response.usage, "prompt_tokens", 0)
        out_tok = getattr(response.usage, "completion_tokens", 0)

        latencies.append(latency)
        in_tokens_list.append(in_tok)
        out_tokens_list.append(out_tok)

        print(f"  Req {i+1:3d}  {latency:.3f}s  in={in_tok:4d}  out={out_tok:3d}  {question[:40]}")

        results.append({
            "request_id": i + 1,
            "question": question,
            "latency_s": round(latency, 4),
            "prompt_tokens": in_tok,
            "completion_tokens": out_tok,
            "isl_target": ISL_TARGET,
            "max_tokens": MAX_TOKENS,
            "host": HOST,
        })

    avg_in = sum(in_tokens_list) / len(in_tokens_list)
    print("\n" + "=" * 65)
    print(f"ISL SUMMARY  target={ISL_TARGET}  actual_avg_in={avg_in:.0f} tokens")
    print("=" * 65)
    print(f"  Requests       : {len(latencies)}")
    print(f"  Avg latency    : {statistics.mean(latencies):.3f} s")
    print(f"  Median latency : {statistics.median(latencies):.3f} s")
    print(f"  Std deviation  : {statistics.stdev(latencies):.3f} s")
    print(f"  Min latency    : {min(latencies):.3f} s")
    print(f"  Max latency    : {max(latencies):.3f} s")
    print(f"  Throughput     : {len(latencies)/sum(latencies):.2f} req/s")
    print(f"  Avg input tok  : {avg_in:.0f}")
    print(f"  Avg output tok : {sum(out_tokens_list)/len(out_tokens_list):.0f}")
    print("=" * 65)

    summary = {
        "isl_target": ISL_TARGET,
        "avg_input_tokens": round(avg_in, 1),
        "max_tokens": MAX_TOKENS,
        "host": HOST,
        "requests": len(latencies),
        "avg_latency_s": round(statistics.mean(latencies), 4),
        "median_latency_s": round(statistics.median(latencies), 4),
        "std_latency_s": round(statistics.stdev(latencies), 4),
        "min_latency_s": round(min(latencies), 4),
        "max_latency_s": round(max(latencies), 4),
        "throughput_rps": round(len(latencies)/sum(latencies), 3),
        "results": results,
    }

    with open(LOG_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[isl_bench] saved to {LOG_FILE}")

if __name__ == "__main__":
    run_isl_benchmark()
