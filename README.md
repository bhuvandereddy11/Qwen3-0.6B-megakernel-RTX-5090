---

# Qwen3-0.6B Megakernel — Independent Reproduction & Analysis

> **Reproduced & Analyzed by:** [@bhuvandereddy11](https://github.com/bhuvandereddy11)
> **Hardware:** RTX 5090 (Vast.ai), PCIe 3.0, Xeon E5-2640 v4, 32GB VRAM
> **Date:** March 11, 2026
> **Original repo:** [emmanuelalo52/Qwen3-0.6B-megakernel-RTX-5090](https://github.com/emmanuelalo52/Qwen3-0.6B-megakernel-RTX-5090)
> **All benchmark data:** [/results](./results)

---

## Summary

We successfully reproduced the megakernel on RTX 5090 and ran 6 additional benchmark angles to better understand its performance characteristics. The kernel shows a **real-world speedup of 1.36x** under single-request conditions, which is a genuine and impressive result. However, our additional testing revealed some important nuances around concurrency and production use cases that are worth noting.

---

## Test 1 — Throughput at Varying Token Lengths (Concurrency=1)

| Max Tokens | Megakernel | vLLM | Speedup |
|---|---|---|---|
| 32 | 15.85 req/s | 11.67 req/s | **1.36x** |
| 64 | 12.40 req/s | 9.80 req/s | **1.27x** |
| 128 | 9.52 req/s | 8.46 req/s | **1.13x** |
| 256 | 8.18 req/s | 7.78 req/s | **1.05x** |

The megakernel advantage is most pronounced at shorter outputs and gradually narrows as token length increases.

---

## Test 2 — Concurrent Requests (Production Scenario)

| System | Throughput (c=4) | Degradation vs c=1 |
|---|---|---|
| Megakernel | 4.18 req/s | -74% |
| vLLM | 7.84 req/s | -33% |

Under concurrent load, vLLM handles multiple requests more efficiently. This is expected given the megakernel is currently designed for single-request inference.

---

## Test 3 — Latency Variance

| Metric | Megakernel | vLLM |
|---|---|---|
| Std deviation | 0.024s | 0.031s |
| Max latency | 0.242s | 0.324s |

The megakernel shows more consistent latency (~25% lower variance) under single-request workloads — a notable strength.

---

## Test 4 — Time To First Token (TTFT / Streaming)

| System | Avg TTFT | p50 | p95 | p99 |
|---|---|---|---|---|
| Megakernel | Not yet supported | — | — | — |
| vLLM | 0.077s | 0.071s | 0.080s | 0.347s |

Streaming support is not yet implemented in the megakernel. For chat interfaces or streaming APIs, this would be an important future addition.

---

## Test 5 — VRAM Usage

| System | VRAM Used | VRAM Free |
|---|---|---|
| Megakernel | 2,030 MB | 30,080 MB |
| vLLM | 29,554 MB | 2,556 MB |

The megakernel is remarkably memory efficient — using only 2GB vs vLLM's 29GB. This leaves significant headroom for other workloads on the same GPU.

---

## Test 6 — Original Claims vs Our Results

| Metric | Original README | Our Results | Notes |
|---|---|---|---|
| Speedup | 2.3x | 1.36x | Likely explained by PCIe 3.0 vs PCIe 5.0 hardware difference |
| Variance | 10x tighter | 25% tighter | Still meaningfully more consistent |
| Streaming | Not mentioned | Not yet supported | Opportunity for future work |
| Concurrency | Not tested | vLLM handles better | Different design goals |

> **Note on the speedup difference:** Our instance used PCIe 3.0 (12 GB/s) while the original benchmark likely used PCIe 5.0 (64 GB/s). The pinned memory transfer bottleneck is much more significant on PCIe 3.0, which likely explains the gap between 2.3x and 1.36x. On matching hardware the original numbers may well hold.

---

## Conclusion

| Scenario | Recommended |
|---|---|
| Single user, short output (<=32 tok) | Megakernel (1.36x faster) |
| Concurrent users | vLLM (more efficient) |
| Streaming / chat UI | vLLM (streaming supported) |
| VRAM constrained environments | Megakernel (14.5x less VRAM) |
| Production API serving | vLLM |

The megakernel is impressive CUDA engineering with a clear architectural advantage for single-request, latency-sensitive workloads. The current implementation is an excellent research foundation. With streaming support and concurrency handling added, it could be very competitive in production scenarios.

All raw data is in [/results](./results). Custom TTFT script: [ttft_benchmark.py](./ttft_benchmark.py).

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