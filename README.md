<div align="center">

<img src="./docs/product/assets/diffusorrt-logo.png" alt="DiffusorRT diffusion-particle logo" width="190">

<h1>DiffusorRT</h1>

<p><strong>Canvas-native inference for diffusion language models.</strong></p>

<p>
Generate, chat, infill, and edit on local hardware through one runtime built for
parallel token refinement, not an autoregressive interface wrapped around a dLLM.
</p>

<p>
<a href="https://www.apache.org/licenses/LICENSE-2.0"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg"></a>
<img alt="Python 3.11" src="https://img.shields.io/badge/python-3.11-3776AB.svg">
<img alt="Apple Silicon / MLX" src="https://img.shields.io/badge/Apple%20Silicon-MLX-000000.svg">
<img alt="NumPy reference" src="https://img.shields.io/badge/reference-NumPy-013243.svg">
<img alt="Status: pre-release" src="https://img.shields.io/badge/status-pre--release-orange.svg">
</p>

<p>
<a href="#why-diffusorrt">Why</a> ·
<a href="#news">News</a> ·
<a href="#features">Features</a> ·
<a href="#supported-models">Models</a> ·
<a href="#getting-started">Getting Started</a> ·
<a href="#architecture">Architecture</a> ·
<a href="#roadmap">Roadmap</a> ·
<a href="#documentation">Docs</a> ·
<a href="#citation">Citation</a> ·
<a href="#license">License</a>
</p>

</div>

> **Repository status.** This public repository is populated by deterministic
> export from the private source mainline, one surface at a time, and each
> export ships with an `EXPORT_RECEIPT.json` naming the source commit and the
> hash of every file. The SDK source and documentation land after the current
> cache-owner cleanup; until then this README describes the product as it
> exists at that source commit, and links into `docs/` resolve once that
> surface is exported.

## Why DiffusorRT

A diffusion language model refines a whole canvas of tokens in parallel. It can
fill a hole in the middle of a document, revise a span, or commit several
positions per step. Serving stacks built for autoregressive decoding hide all of
that behind a left-to-right stream.

DiffusorRT makes the canvas the first-class object:

- **Native canvas operations.** `generate`, multi-turn `chat`, bidirectional
  `infill`, and span `edit` share one Engine and one runtime session.
- **Exact work reduction.** Active-window execution forwards only the rows
  that still need evidence and keeps the full forward as its oracle. Reduced
  execution reaches the product only after it matches the oracle token for
  token.
- **Explicit K/V semantics.** Prompt and frozen-block reuse bind identity,
  validity, lifecycle, and publication to a single cache owner instead of
  trusting a cache pointer.
- **Honest measurement.** DiffusorRT does not claim that diffusion models beat
  autoregressive models in tokens per second. Every performance statement names
  its model, hardware, precision, and cache state.

## News

- **2026-09-01 — Exact-D2F now runs on the native K/V manager.** The Apple/MLX
  product path drives the compiled cache manager as its only persistent K/V
  owner, with admission before allocation, transactional publication, and
  checked shutdown through `Engine.close()`. The installed product evidence
  binds the Apache-2.0 SDK and the separately licensed runtime wheel to the
  same source identity.
  [Details →](./docs/product/kv-cache-manager-v1.md)
- **2026-08-24 — The private CUDA provider executes real Dream BF16 windows.**
  The provider built and installed on Linux, preserved the 11-symbol callable
  surface, and passed the seven-window numerical gate on A100 and L40. The
  public runtime still refuses CUDA; this is a foundation, not a product claim.
  [Details →](./docs/product/cuda-runtime-v1.md)
- **2026-08-20 — Qwen3-MDLM v0.1 completed product admission.** The exact 0.6B
  revision passed reproducible parity and capability gates and the installed
  `doctor`, `pull`, `list`, `run`, and `serve` workflow on Apple M4 Max with
  MLX fp32.
  [Details →](./docs/product/model-coverage-v1.md)
- **2026-08-15 — The native C++ boundary runs a real Dream window.** The
  callable runtime builds as a binary wheel and agrees with the Python/MLX
  oracle on row order, argmax, and top-5.
  [Details →](./docs/product/callable-runtime-v1.md)
- **2026-08-13 — One command reaches a model from an empty cache.** `run`,
  `pull`, `list`, `doctor`, and `serve` accept the pinned model ID and complete
  acquisition, compilation, loading, and generation.
  [Details →](./docs/product/client-runtime-v1.md)

See [Product Updates](./docs/product/CHANGELOG.md) for completed milestones,
removals, compatibility notes, and scope limits.

## Features

| | |
|---|---|
| **Canvas-native generation** | One Engine exposes `generate`, multi-turn `chat`, bidirectional `infill`, and span `edit`. |
| **Exact work reduction** | Active-window D2F executes only rows that still need evidence and keeps the full forward as its oracle. |
| **One K/V owner** | A native cache manager owns identity, capacity, leases, transactions, and a versioned operation trace; the Python manager is its differential oracle. |
| **Model-addressable workflow** | `doctor`, `pull`, `run`, `list`, and `serve` resolve the supported model without a package path. |
| **Execution packages** | The Reference Compiler pins source identity, checks disk cost, converts into a runtime package, and publishes a receipt. |
| **Portable contracts** | NumPy is the executable reference; MLX is the accelerated Apple path; native runtimes attach through a stable ABI. |

## Supported models

| Model | Status | Admitted profile |
|---|---|---|
| `Dream-org/Dream-v0-Instruct-7B` | Product-supported at its pinned revision | Apple Silicon / MLX; the performance anchor |
| `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1` | Product-supported at its pinned revision | Apple M4 Max / MLX fp32 |
| LLaDA-8B, Dream-Coder | Candidates with parity evidence | Not product-supported |

Support means the exact revision passed its model-specific product gate on named
hardware. [Model Coverage v1](./docs/product/model-coverage-v1.md) defines the
ladder from candidate to product-supported and records every admission.

## Getting Started

### Requirements

- Python 3.11;
- Apple Silicon for the accelerated MLX path;
- free storage for the selected package (Qwen is about 3.0 GB; pinned Dream
  is a roughly 47 GiB first-run path);
- explicit consent when using pinned Dream's custom tokenizer code.

### Two layers, one runtime

This repository is the Apache-2.0 SDK: the Python API and CLI, the Reference
Compiler, the NumPy reference runtime, schemas, public C headers, conformance
tools, and documentation. The exact active-window profile on Apple Silicon runs
through a **separately licensed native runtime wheel**. Without that wheel,
DiffusorRT refuses the exact profile with an actionable error; it never
substitutes a different cache silently. The reference and conformance paths
need only this repository.

### Install from source

No stable wheel has been published yet.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e "./python[compiler,mlx-local,serve]"
```

### Check, prepare, and run

```bash
MODEL="dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1"

# Inspect support, cache state, trust requirements, and disk cost.
diffusorrt doctor "$MODEL"

# One command acquires, compiles, loads, and generates.
diffusorrt run "$MODEL" --prompt "Explain diffusion language models in one paragraph."

# Prepare without generating. Add --offline to forbid network access.
diffusorrt pull "$MODEL"

# Start the local foreground service with /health and /v1/generate.
diffusorrt serve "$MODEL"
```

### Python API

```python
from diffusor_rt import Engine

with Engine.from_pretrained(
    "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1",
    backend="auto",
) as engine:
    result = engine.generate(
        "Explain diffusion language models in one paragraph.",
        max_tokens=64,
    )
    print(result.text)
```

Prepared execution packages stay available for offline development and
conformance through `Engine.from_package()`.

## Architecture

<p align="center">
<img src="./docs/product/assets/architecture.svg" alt="DiffusorRT architecture: compiler, package, engine, canvas runtime, and three backends" width="880">
</p>

The compiler owns acquisition and package publication. The runtime owns decode
semantics, cache lifecycle, and the operation trace. Backends own tensor math,
device memory, and synchronization. A native runtime attaches through the
stable C ABI and must either serve the requested profile or refuse it; it may
not change semantics.

```text
model snapshot -> Reference Compiler -> execution package
  -> Engine / runtime session
    -> canvas + active-window planner + exact K/V manager
      -> NumPy reference | MLX | native runtime
```

## Roadmap

Work proceeds in three lanes under one integration authority, so no lane can
create a second runtime or a second control plane.

| Lane | Next |
|---|---|
| **Model coverage** | Qwen3-BD3LM; a bounded SDAR / Fast-dLLM-v2 selection spike; LLaDA-8B and Dream-Coder toward product support. |
| **Runtime and K/V** | Retire the quiesced legacy cache owners; record the installed real-Dream manager trace with byte accounting. |
| **CUDA** | Versioned BF16, INT8, and FP8 precision profiles behind the private provider, then public runtime negotiation and product admission. |
| **Then** | Packed active-window feasibility, a continuous scheduler and service, and a signed developer preview. |

Speculative drafting with an autoregressive verifier is an optional execution
mode on the roadmap, not the runtime's identity.

## Documentation

- [Documentation index](./docs/README.md) — user, integration, and conformance guides
- [Product charter](./docs/product/README.md) — identity, platform boundaries, and positioning
- [Product Updates](./docs/product/CHANGELOG.md) — recent changes and completed milestones
- [Client Runtime v1](./docs/product/client-runtime-v1.md) — model resolution, trust, preparation, and reporting
- [Callable Runtime v1](./docs/product/callable-runtime-v1.md) — C lifecycle, ownership, errors, and execution boundary
- [Model Coverage v1](./docs/product/model-coverage-v1.md) — pinned model identity, parity, capability, and product admission
- [KV Cache Manager v1](./docs/product/kv-cache-manager-v1.md) — exact cache ownership, identity, capacity, and transactions
- [CUDA Runtime v1](./docs/product/cuda-runtime-v1.md) — planned NVIDIA backend contract and hardware gates
- [Benchmarks](./docs/benchmarks.md) — benchmark shapes, prompts, and reporting fields
- [Conformance](./docs/conformance.md) — adapter, package, and generation conformance

## Contributing

Read [Contributing](./CONTRIBUTING.md) for development setup and change
requirements, and [Security](./SECURITY.md) for how to report a vulnerability.
Public contributions land in the private source mainline first and return
through the deterministic exporter, so the public tree never becomes a second
implementation.

## Citation

```bibtex
@software{diffusorrt2026,
  title  = {DiffusorRT: Canvas-native inference for diffusion language models},
  author = {Li, Jimmy Yilong},
  year   = {2026},
  url    = {https://github.com/jimmy-yilong-li/DiffusorRT}
}
```

## License

DiffusorRT provides an **open-source SDK and executable reference runtime**
under the [Apache License 2.0](./LICENSE). The public surface includes the
Python API and CLI, Reference Compiler, NumPy runtime, schemas, public C
headers, documentation, and conformance assets.

Optimized native runtimes and device-specific backends ship as separately
licensed proprietary binaries; their source is not part of the Apache-2.0
release. Model weights, tokenizers, and compiled model artifacts keep their
upstream terms.

See [Copyright, Licensing, and Distribution Policy](./COPYRIGHT_AND_LICENSING.md)
for the complete source and binary boundary.
