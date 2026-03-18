# Qwen3-0.6B Megakernel — T4 Port & Benchmarking

> **Author:** Bhuvan Dereddy (FloTorch)
> **Hardware:** AWS EC2 g4dn — Tesla T4, sm_75, 40 SMs, 15GB VRAM
> **CUDA:** 13.0 | **PyTorch:** 2.10.0+cu130 | **Date:** March 13, 2026
> **Original kernel:** [emmanuelalo52/Qwen3-0.6B-megakernel-RTX-5090](https://github.com/emmanuelalo52/Qwen3-0.6B-megakernel-RTX-5090)
> **All benchmark data:** [/T4testing](./T4testing)

---

## What This Repo Does

Emmanuel Alo authored a custom CUDA megakernel for Qwen3-0.6B inference, originally hardcoded exclusively for the RTX 5090 (sm_120, 170 SMs, 100KB shared memory, Blackwell-only CUDA instructions). This repo ports that kernel to run on a Tesla T4 GPU and benchmarks it against vLLM across multiple ISL and OSL configurations.

---

## Porting Changes Made (RTX 5090 → Tesla T4)

| File | Change | RTX 5090 | Tesla T4 |
|---|---|---|---|
| `megakernel_5090.cu` line 13 | Architecture guard | sm_80 | sm_75 |
| `megakernel_5090.cu` line 31 | LDG_NUM_BLOCKS | 170 | 40 |
| `megakernel_5090.cu` line 39 | LDG_LM_NUM_BLOCKS | 680 | 160 |
| `megakernel_5090.cu` line 128 | Spin-wait instruction | nanosleep.u32 256 (sm_80+ only) | for loop with asm volatile |
| `megakernel_5090.cu` lines 1409/1416 | Shared memory limit | 100 * 1024 (100KB) | 64 * 1024 (64KB) |
| `setup.py` line 34 | NVCC arch flag | -arch=sm_120 | -arch=sm_75 |

---

## Environment Setup
```bash
git clone https://github.com/bhuvandereddy11/Qwen3-0.6B-megakernel-RTX-5090.git
cd Qwen3-0.6B-megakernel-RTX-5090
git checkout t4-port

python3 -m venv venv_t4
source venv_t4/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install ninja setuptools numpy fastapi uvicorn pydantic transformers accelerate orjson

export PATH=/usr/local/cuda-13.0/bin:$PATH
export CUDA_HOME=/usr/local/cuda-13.0
python setup.py build_ext --inplace
```

---

## Running the Servers

**Megakernel (port 8000):**
```bash
PYTHONPATH=/path/to/repo python Tools/megakernel/megakernel.py
```

**vLLM baseline (port 8001):**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-0.6B \
  --port 8001 \
  --dtype float16 \
  --max-model-len 4096
```

**Running benchmarks:**
```bash
# OSL benchmark
cat > .env << 'EOF'
HOST=http://localhost:8000
MODEL=Qwen/Qwen3-0.6B
MAX_TOKENS=128
EOF
python client_benchmark.py --requests 100

# ISL benchmark
cat > .env << 'EOF'
HOST=http://localhost:8000
MODEL=Qwen/Qwen3-0.6B
MAX_TOKENS=128
ISL_TARGET=500
LOG_FILE=results.json
EOF
python isl_benchmark.py
```

---

## Benchmark Results

All tests: 100 requests, concurrency=1 (sequential), greedy decoding (temperature=0.0), Qwen/Qwen3-0.6B float16, Tesla T4.

---

### Test 1 — OSL Sweep (Output Sequence Length)

**Setup:** Input fixed at ~20 tokens (short factual questions). Max output tokens varied.

| OSL | Megakernel Avg | Megakernel Median | Megakernel Throughput | vLLM Avg | vLLM Median | vLLM Throughput | Winner |
|---|---|---|---|---|---|---|---|
| 32 tokens | 0.340s | 0.276s | 2.94 req/s | 0.167s | 0.215s | 5.99 req/s | vLLM by 2.04x |
| 64 tokens | 0.292s | 0.303s | 3.43 req/s | 0.258s | 0.269s | 3.88 req/s | vLLM by 1.13x |
| 128 tokens | 0.357s | 0.303s | 2.80 req/s | 0.340s | 0.276s | 2.94 req/s | vLLM by 1.05x |
| 256 tokens | 0.419s | 0.303s | 2.39 req/s | 0.405s | 0.271s | 2.47 req/s | vLLM by 1.03x |

**Finding:** The gap narrows as output length increases. At 256 tokens both systems are essentially equivalent. The megakernel's advantage at short outputs disappears at longer generations.

---

### Test 2 — ISL Sweep (Input Sequence Length)

**Setup:** Output capped at 128 tokens. Input length increased by prepending padding context to each prompt.

| ISL Target | Actual Avg Input | Megakernel Avg | Megakernel Throughput | vLLM Avg | vLLM Throughput | Winner |
|---|---|---|---|---|---|---|
| ~20 tokens | 20 tokens | 0.355s | 2.82 req/s | 0.337s | 2.97 req/s | vLLM by 1.05x |
| ~100 tokens | 55 tokens | 0.393s | 2.55 req/s | 0.203s | 4.93 req/s | vLLM by 1.93x |
| ~500 tokens | 650 tokens | 3.438s | 0.29 req/s | 0.152s | 6.60 req/s | vLLM by 22.6x |
| ~1000 tokens | 1280 tokens | 7.841s | 0.13 req/s | 0.158s | 6.35 req/s | vLLM by 49.6x |

**Critical finding:** The megakernel degrades severely with longer inputs. At 1280 input tokens it takes 7.8s vs vLLM's 0.16s — nearly 50x slower. The prefill path processes each token sequentially through the full kernel loop and was never optimized for long-context inputs. vLLM actually gets faster with longer inputs because the model produces shorter, more concise answers.

---

## Overall Summary

| Scenario | Megakernel T4 | vLLM T4 | Verdict |
|---|---|---|---|
| OSL=32, ISL=20 | 0.340s / 2.94 req/s | 0.167s / 5.99 req/s | vLLM 2x faster |
| OSL=64, ISL=20 | 0.292s / 3.43 req/s | 0.258s / 3.88 req/s | vLLM 1.13x faster |
| OSL=128, ISL=20 | 0.357s / 2.80 req/s | 0.340s / 2.94 req/s | vLLM 1.05x faster |
| OSL=256, ISL=20 | 0.419s / 2.39 req/s | 0.405s / 2.47 req/s | Essentially equal |
| ISL=55, OSL=128 | 0.393s / 2.55 req/s | 0.203s / 4.93 req/s | vLLM 2x faster |
| ISL=650, OSL=128 | 3.438s / 0.29 req/s | 0.152s / 6.60 req/s | vLLM 22x faster |
| ISL=1280, OSL=128 | 7.841s / 0.13 req/s | 0.158s / 6.35 req/s | vLLM 50x faster |

The T4 port is a working proof of concept — the kernel compiles, loads, and runs correctly on T4 with only 6 code changes. However vLLM outperforms it across all tested configurations on this hardware. The prefill path is the most critical bottleneck for real-world use cases with longer inputs.

---

## Key Observations

**OSL (decode performance):** The megakernel's single-kernel-launch design reduces CPU round trips between transformer layers. On RTX 5090 (170 SMs) this gave a 2.3x speedup. On T4 (40 SMs) the benefit is smaller — only ~2x at OSL=32 and disappears by OSL=256.

**ISL (prefill performance):** This is where the megakernel falls significantly behind. The prefill implementation runs each input token through the full kernel sequentially. At 1280 input tokens this takes 7.8 seconds — making it impractical for any real-world RAG, document QA, or long-context use case.

**vLLM ISL behavior:** vLLM gets faster as input grows because the model answers more concisely with more context. Average output tokens drops from ~50 (ISL=20) to ~19 (ISL=1280), reducing the decode time.

---

## Next Steps

- Optimize the prefill path — batched prefill instead of sequential per-token processing
- Profile with Nsight Compute on bare-metal T4 (shared GPU blocks hardware counters)
- Collaborate with Emmanuel Alo on GPU-agnostic SM count configuration
- Test on A10G (sm_86, 72 SMs) as a middle ground between T4 and RTX 5090

---

## File Structure
```
T4testing/
  megakernel_t4_32tok.json     OSL=32 tokens, ISL~20, megakernel
  megakernel_t4_64tok.json     OSL=64 tokens, ISL~20, megakernel
  megakernel_t4_128tok.json    OSL=128 tokens, ISL~20, megakernel
  megakernel_t4_256tok.json    OSL=256 tokens, ISL~20, megakernel
  megakernel_t4_isl20.json     ISL~20 tokens, OSL=128, megakernel
  megakernel_t4_isl100.json    ISL~55 tokens, OSL=128, megakernel
  megakernel_t4_isl500.json    ISL~650 tokens, OSL=128, megakernel
  megakernel_t4_isl1000.json   ISL~1280 tokens, OSL=128, megakernel
  vllm_t4_*.json               Matching vLLM baselines for all above
Tools/megakernel/              Modified kernel source (T4 port)
  megakernel_5090.cu           Main CUDA kernel (modified for T4)
  megakernel.py                FastAPI inference server
  qwen_ops.cpp                 PyTorch C++ extension bridge
client_benchmark.py            OSL benchmark script
isl_benchmark.py               ISL benchmark script
setup.py                       Build script (sm_75)
```
