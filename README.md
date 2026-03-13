# Qwen3-0.6B Megakernel — Independent Reproduction & Critical Analysis

> **Reproduced & Analyzed by:** [@bhuvandereddy11](https://github.com/bhuvandereddy11)  
> **Hardware:** RTX 5090 (Vast.ai), PCIe 3.0, Xeon E5-2640 v4, 32GB VRAM  
> **Date:** March 11, 2026  
> **Original repo:** [emmanuelalo52/Qwen3-0.6B-megakernel-RTX-5090](https://github.com/emmanuelalo52/Qwen3-0.6B-megakernel-RTX-5090)  
> **All benchmark data:** [/results](./results)

---

## TL;DR — The 2.3x Claim Does Not Hold

After 6 independent benchmark angles on identical hardware (RTX 5090), the megakernel shows a **real-world speedup of 1.36x at best** and is **outperformed by vLLM in every production scenario**.

---

## Test 1 — Throughput at Varying Token Lengths (Concurrency=1)

| Max Tokens | Megakernel | vLLM | Speedup |
|---|---|---|---|
| 32 | 15.85 req/s | 11.67 req/s | **1.36x** |
| 64 | 12.40 req/s | 9.80 req/s | **1.27x** |
| 128 | 9.52 req/s | 8.46 req/s | **1.13x** |
| 256 | 8.18 req/s | 7.78 req/s | **1.05x** |

The megakernel advantage vanishes as output length increases. At 256 tokens it is essentially identical to vLLM.

---

## Test 2 — Concurrent Requests (Production Scenario)

| System | Throughput (c=4) | Degradation vs c=1 |
|---|---|---|
| Megakernel | 4.18 req/s | -74% |
| vLLM | 7.84 req/s | -33% |

Under concurrent load, **vLLM beats the megakernel by 1.88x**. The megakernel collapses under any parallelism.

---

## Test 3 — Latency Variance

| Metric | Megakernel | vLLM |
|---|---|---|
| Std deviation | 0.024s | 0.031s |
| Max latency | 0.242s | 0.324s |

Megakernel is ~25% more consistent at single-request workloads. The README claim of "10x tighter" is overstated by 8x.

---

## Test 4 — Time To First Token (TTFT / Streaming)

| System | Avg TTFT | p50 | p95 | p99 |
|---|---|---|---|---|
| Megakernel | NOT SUPPORTED | — | — | — |
| vLLM | 0.077s | 0.071s | 0.080s | 0.347s |

**Critical: The megakernel has zero streaming support.** Users wait for the full response before seeing any output. In any real chatbot or API product this is a fatal flaw. vLLM streams token-by-token with p95 TTFT of 80ms.

---

## Test 5 — VRAM Usage

| System | VRAM Used | VRAM Free |
|---|---|---|
| Megakernel | 2,030 MB | 30,080 MB |
| vLLM | 29,554 MB | 2,556 MB |

vLLM pre-allocates ~27GB for KV cache enabling 123 concurrent requests. The megakernel uses only 2GB but cannot use the remaining 30GB for batching or caching.

---

## Test 6 — Claimed vs Reproduced

| Metric | README Claim | Reproduced | Verdict |
|---|---|---|---|
| Speedup | 2.3x | 1.36x | OVERSTATED by 70% |
| Variance | 10x tighter | 25% tighter | OVERSTATED by 8x |
| Streaming | Not mentioned | Not supported | OMITTED |
| Concurrency | Not tested | vLLM wins 1.88x | OMITTED |

---

## Conclusion

| Scenario | Winner |
|---|---|
| Single user, short output (<=32 tok) | Megakernel (1.36x) |
| Concurrent users | vLLM (1.88x) |
| Streaming / chat UI | vLLM only |
| VRAM efficiency | Megakernel (14.5x less) |
| Production readiness | vLLM |

The megakernel is impressive CUDA engineering but is a research prototype. The 2.3x claim only holds at 32 tokens, single-request, on better hardware (PCIe 5.0). It is not reproducible in production conditions.

All raw data is in [/results](./results). Custom TTFT script: [ttft_benchmark.py](./ttft_benchmark.py).

---
---
---

# Original README (preserved for reference)

> The following is the original content from emmanuelalo52.

---

# Qwen3-0.6B Megakernel — Reproduction Guide

Custom CUDA megakernel for Qwen3-0.6B inference on RTX 5090, benchmarked against vLLM standard PagedAttention baseline.

## Original Benchmark Results (RTX 5090, float16, 32 max tokens)

| Metric | Megakernel | vLLM (enforce-eager) | Speedup |
|---|---|---|---|
| Avg latency | 0.052s | 0.120s | **2.3x** |
| Median latency | 0.052s | 0.154s | **2.9x** |
| Tokens/sec | 606.7 | 266.5 | **2.3x** |
| Req/s | 17.04 | 8.33 | **2.0x** |
| Variance (min to max) | 0.048-0.058s | 0.048-0.162s | 10x tighter |

## Requirements

- NVIDIA RTX 5090 (sm_120 architecture)
- CUDA 13.0 driver
- Python 3.12
- Ubuntu 24

## Key Design Decisions

**Single kernel launch per inference phase** — runs all decode steps inside one launch using GPU-side grid barriers.

**Partial KV cache reset** — only zeros positions actually written instead of clearing the full cache.

**Single cudaStreamSynchronize per request** — all token generation steps run on-device with no CPU sync until final output.

**Pinned memory output buffer** — output token log backed by pinned CPU memory for fast DMA transfer.
