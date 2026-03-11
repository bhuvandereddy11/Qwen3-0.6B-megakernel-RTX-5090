# Qwen3-0.6B Megakernel — Independent Reproduction & Analysis

> **Reproduced by:** bhuvandereddy11  
> **Hardware:** RTX 5090 (Vast.ai), PCIe 3.0, Xeon E5-2640 v4, 32GB VRAM  
> **Date:** March 11, 2026  
> **Original repo:** [emmanuelalo52/Qwen3-0.6B-megakernel-RTX-5090](https://github.com/emmanuelalo52/Qwen3-0.6B-megakernel-RTX-5090)

---

## Summary

Independent reproduction and multi-angle analysis of the megakernel performance claims. The original README claims a **2.3x speedup** over vLLM. After benchmarking across 6 test conditions, the real picture is more nuanced.

**The megakernel works and is faster than vLLM — but only in specific conditions, and has critical production limitations.**

---

## 1. Throughput — Single Request (Concurrency=1)

| Max Tokens | Megakernel | vLLM | Speedup |
|---|---|---|---|
| 32 | 15.85 req/s | 11.67 req/s | **1.36x** |
| 64 | 12.40 req/s | 9.80 req/s | **1.27x** |
| 128 | 9.52 req/s | 8.46 req/s | **1.13x** |
| 256 | 8.18 req/s | 7.78 req/s | **1.05x** |

Megakernel advantage shrinks rapidly as output length increases.

---

## 2. Concurrency Test (Concurrency=4)

| System | Throughput | Degradation vs c=1 |
|---|---|---|
| Megakernel | 4.18 req/s | -74% |
| vLLM | 7.84 req/s | -33% |

**vLLM beats megakernel by 1.88x at concurrency=4.**

---

## 3. Latency Variance

| Metric | Megakernel | vLLM |
|---|---|---|
| Std deviation | 0.024s | 0.031s |
| Max latency | 0.242s | 0.324s |

Megakernel is ~25% more consistent at single-request workloads.

---

## 4. Time To First Token (TTFT) — Streaming

| System | Avg | p50 | p95 | p99 |
|---|---|---|---|---|
| Megakernel | ❌ No streaming | - | - | - |
| vLLM | 0.077s | 0.071s | 0.080s | 0.347s |

**Critical: Megakernel has no streaming support.** Users wait for full response before seeing any output.

---

## 5. VRAM Usage

| System | VRAM Used | VRAM Free |
|---|---|---|
| Megakernel | 2,030 MB | 30,080 MB |
| vLLM | 29,554 MB | 2,556 MB |

vLLM uses 14.5x more VRAM but allocates it for KV cache enabling 123x concurrent requests.

---

## 6. Claim vs Reality

| Metric | README Claim | Reproduced | Verdict |
|---|---|---|---|
| Speedup | 2.3x | 1.36x | ❌ Overstated |
| Variance | 10x tighter | ~25% tighter | ❌ Overstated |
| Streaming | Not mentioned | Not supported | ❌ Missing |
| Concurrency | Not tested | vLLM wins 1.88x | ❌ Missing |

---

## Conclusion

| Scenario | Winner |
|---|---|
| Single user, short output | Megakernel (1.36x) |
| Concurrent users | vLLM (1.88x) |
| Streaming / chat UI | vLLM only |
| VRAM efficiency | Megakernel (14.5x less) |
| Production readiness | vLLM |

The megakernel is an impressive CUDA engineering achievement but is a research prototype, not a production system. The 2.3x claim is only achievable on better hardware (PCIe 5.0) under single-request conditions not representative of real workloads.
