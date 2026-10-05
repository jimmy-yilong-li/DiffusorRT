<div align="center">

<img src="./docs/product/assets/diffusorrt-logo.png" alt="DiffusorRT" width="190">

<h1>DiffusorRT</h1>

<p><strong>A Python runtime for diffusion language models on local hardware.</strong></p>

<p>
<a href="#news">News</a> ·
<a href="#install">Install</a> ·
<a href="#python-api">Python API</a> ·
<a href="#command-line">Command line</a> ·
<a href="#model-manager-and-local-service">Model manager</a> ·
<a href="#supported-models-and-platforms">Supported models</a> ·
<a href="#license">License</a>
</p>

</div>

DiffusorRT provides model loading, generation, conversation history, context
management, and a local HTTP service through one importable library. Models use
the same loader and runtime; the command line and optional TUI call that library.

This is the user-facing product repository: installation, downloadable releases,
examples and the selected open-source scope. Collaborator development snapshots
and beta features are not automatically part of a public release or support
claim. Use the delivered version and its documented capabilities.

**0.2.1 is a Developer Preview, released in stages.** This batch provides
an installable Apache-2.0 Python wheel and a small selection of API source files,
not the full development repository. Optimized native Apple/CUDA backends and
model weights are separate and are not included in this wheel.

## News

**2026-10-05 — Maintenance development update; not released.** Source fixes
correct BD3 round allocation for longer requests, avoid forwarding future
blocks, and improve checked MLX buffer release. Failed cleanup remains retryable
and blocks new work; unsupported infill/edit now has a capability explanation.
Evaluation tools correct answer parsing and duplicate-question sampling and
retain actual step budgets and stage timings. These changes do not establish
an accuracy or latency improvement. They are **not in the 0.2.1 wheel below**;
the next maintenance wheel will have a new version and separate validation.

**2026-10-03 — SDK 0.2.1 correctness update.**
[Download the preview](https://github.com/jimmy-yilong-li/DiffusorRT/releases/tag/v0.2.1).
BD3 now completes the current diffusion block before stopping at EOS, avoiding
the unresolved positions that caused empty or incomplete replies in `0.2.0`.
Newly prepared packages also retain the model context limit needed by durable
chat. Explicit weight-hash verification includes sharded payloads. The release
includes the Apache SDK wheel, standalone example and checksums, not a private
native backend or the full source tree. Lightweight installed checks passed;
this maintenance release does not renew all real-model certifications or claim
a latency improvement. Finishing a BD3 block can require more forward passes
than the incorrect early stop.

**2026-10-02 — Development update; not yet released.** Source improvements
strengthen sharded-weight verification and preserve model context limits.
The Python MLX exact-K/V path also removes an intermediate read copy, reuses
forward-local masks, avoids per-fragment heap collection and completes capture
copies once per world. Native ownership and exact FP32 cache storage are
unchanged. The native K/V improvements require a separate compatible private
Apple wheel; the public BD3 example remains cache-free. The separate native
binary is not delivered by the `0.2.1` SDK update. No general generation-speed
or new model/platform claim follows.

**2026-09-15 — Developer Preview available.**
[v0.2.0](https://github.com/jimmy-yilong-li/DiffusorRT/releases/tag/v0.2.0)
provides an installable Apache Python wheel, quickstart and checksums. Follow
the installation instructions below; no source build is required.

## Install

Use **Python 3.11**. The accelerated example below needs **Apple Silicon and
macOS**. It uses Qwen3-BD3LM 0.6B and does not require a private backend wheel.

Install directly from the GitHub release; no Git clone or compiler build is
needed to install the Python library:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install "diffusor-rt[compiler,mlx-local] @ https://github.com/jimmy-yilong-li/DiffusorRT/releases/download/v0.2.1/diffusor_rt-0.2.1-py3-none-any.whl"
```

An existing Python 3.11 conda environment works too; skip the first two commands
when it is already active. The `compiler` extra installs the dependencies for
downloading and preparing models, and `mlx-local` installs the Apple backend.

Alternatively, download the wheel from [Release assets](https://github.com/jimmy-yilong-li/DiffusorRT/releases/tag/v0.2.1)
and install it locally:

```bash
python -m pip install "./diffusor_rt-0.2.1-py3-none-any.whl[compiler,mlx-local]"
```

For only the base API, package tools, and NumPy reference runtime, omit the
extras. This does not install an accelerated backend or model-download tools:

```bash
python -m pip install "https://github.com/jimmy-yilong-li/DiffusorRT/releases/download/v0.2.1/diffusor_rt-0.2.1-py3-none-any.whl"
```

The package name is `diffusor-rt`; the Python import is `diffusor_rt`.
This preview is distributed through GitHub Releases, not PyPI. The
`py3-none-any` tag describes the Python wheel, not GPU or model support on every
platform.

To upgrade an existing installation, use the same install command with
`--upgrade`. Model-ID loading rebuilds stale compiled packages automatically;
you do not need to erase model downloads or conversation history. Explicit
package paths stay pinned to their contents, so prepare a new package with
`pull` to obtain corrected metadata. The older `0.2.0` assets remain available,
but that version has the BD3 early-stop and chat-metadata defects fixed here.

If an older attempt left a chat session bound to the rebuilt package, published
`0.2.1` may report an execution-identity mismatch. Keep its history by choosing
a new `--session` name. Use `diffusorrt chat --list-sessions` to inspect saved
sessions; delete an old session only when you intend to discard it. Automatic
recovery of empty sessions is an unreleased source correction, not part of the
wheel linked here.

The published preview still has the older long-output step allocation and MLX
allocator-cache release behavior. A larger output cap may change answer quality;
do not assume `close()` empties the allocator cache in that version. The small
default checkpoint can give incorrect facts or arithmetic. Successful generation
is not a correctness guarantee for its answers.

Verify the installation without loading a model:

```bash
python -c "import diffusor_rt; print(diffusor_rt.__version__)"
python -m diffusor_rt --help
```

## Python API

```python
from diffusor_rt import Engine

with Engine.from_pretrained(
    "dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1",
    backend="mlx_local",
) as engine:
    result = engine.generate(
        "Hello!",
        max_tokens=32,
    )
    print(result.text)
```

The first run downloads about **1.5 GB** of weights and builds a roughly
**3 GB** execution package. Allow additional room for dependencies and temporary
build files. `doctor` reserves about **8.9 GiB** for preparation, including
temporary-build headroom; this is not the final cache size (about **4.2 GiB**).
Later runs reuse the package. Model preparation and generation are
real compute workloads; importing the library and displaying help are not.

Supported model IDs resolve to their pinned revisions automatically. No TOML or
manual package construction is needed for the models listed below. The context
manager closes the engine and releases its resources.

To use an already prepared execution package without network access:

```python
from diffusor_rt import Engine

with Engine.from_pretrained(
    "/path/to/prepared-package",
    backend="mlx_local",
    offline=True,
) as engine:
    print(engine.generate("Hello!", max_tokens=16).text)
```

A prepared package is a DiffusorRT execution package, not an arbitrary directory
of Hugging Face weights. `offline=True` can also use previously cached source.

The same API is available as a ready-to-run script:

```bash
# From this repository, after installing the wheel:
python examples/quickstart.py --prompt "Hello!" --max-tokens 32
```

Download [quickstart.py](./examples/quickstart.py) separately if you do not want
to clone the repository. It is a thin API example, not a second runtime.

## Command line

```bash
MODEL="dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1"

# Inspect the platform and cache without running generation.
diffusorrt doctor "$MODEL"

# Download and prepare once; subsequent runs reuse the package.
diffusorrt pull "$MODEL"

diffusorrt run "$MODEL" --prompt "Hello!" --max-tokens 32
diffusorrt chat "$MODEL" --session local-demo
```

`run` can prepare an uncached model itself; `pull` is optional. Use `--offline`
to forbid network access after caching. Inspect each command with `--help` for
supported runtime parameters.

Before `pull`, `doctor` exits with status 1 when the model is not prepared;
follow its preparation instruction. This does not mean installation failed.
`chat` reuses the prepared model and resumes the named conversation on later
invocations. A generation token limit can still truncate a long answer; it is
not a quality guarantee.

Conversation history is stored locally in **plain, unencrypted SQLite**.
Context management keeps requests within the model's token budget. Summaries
are opt-in and require additional model work. DiffusorRT is a framework, not a
fixed chatbot: applications can provide their own `MemoryProvider`; there is no
automatic long-term memory collection or retrieval by default.

## Model manager and local service

Install the optional TUI and HTTP service dependencies:

```bash
python -m pip install "diffusor-rt[compiler,mlx-local,serve,tui] @ https://github.com/jimmy-yilong-li/DiffusorRT/releases/download/v0.2.1/diffusor_rt-0.2.1-py3-none-any.whl"
```

The TUI manages model packages and saved run profiles:

```bash
diffusorrt models

# After saving a profile named daily in the TUI:
diffusorrt run --run-profile daily --prompt "Explain dLLMs briefly."
diffusorrt chat --run-profile daily --session research-notes
```

Start a foreground HTTP service on localhost:

```bash
diffusorrt serve "dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1" \
  --host 127.0.0.1 --port 8000
```

From another terminal:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain dLLMs briefly.","max_tokens":16}'
```

Press Ctrl-C to close the service. Keep it on localhost: this preview is not an
authenticated Internet-facing service and does not provide OpenAI-compatible
streaming or continuous batching.

## Supported models and platforms

| Model | This public wheel |
|---|---|
| `dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1` | Recommended first run. Apple/MLX fp32, ordinary prompt lengths, fixed-quota decoding, no K/V cache; deterministic sampling (`temperature=0`, `top_p=1`). |
| `GSAI-ML/LLaDA-8B-Instruct` | Apple/MLX fp32 baseline. Much larger: the execution package is about 32 GB. |
| Dream-7B and Qwen3-MDLM | Their admitted exact-D2F profiles also need a compatible private native runtime, which is not included in this release. |

The retained real-model baseline for the two Apache-only profiles names
**Apple M4 Max / 64 GB / MLX fp32**. This SDK patch does not replace that evidence
with a new fixed-source acceptance for every model. Other Apple hardware is not
independently certified by this release. NumPy remains a portable reference,
not a claim of practical 8B CPU
performance. CUDA, reduced-precision BD3, BD3 caching, and mobile acceleration
are not provided by this public wheel.

The library exposes generation, conversation, infill, and edit interfaces;
availability depends on the selected model's decoding contract. In particular,
BD3 does not support infill. Unsupported combinations fail explicitly.

## Release and verification

This first source batch contains only the public import/module entry points,
the Engine API, and the quickstart example:

- [Python API entry point](./python/src/diffusor_rt/__init__.py)
- [Module command entry point](./python/src/diffusor_rt/__main__.py)
- [Engine API](./python/src/edllm/engine.py)
- [Quickstart example](./examples/quickstart.py)

These selected files are for reading and integration examples, not a complete
editable source installation. Install the wheel to run the library. The wheel
necessarily contains the public Python modules and resources it needs at
runtime; a Python wheel is not an encrypted or closed-source binary. The
remaining source tree, native implementations, tests, research, and internal
documents are not part of this repository update.

The [0.2.1 release](https://github.com/jimmy-yilong-li/DiffusorRT/releases/tag/v0.2.1)
contains the wheel, standalone example, and `SHA256SUMS`. Downloaded files can be
checked on macOS with `shasum -a 256 -c SHA256SUMS`, or on Linux with
`sha256sum -c SHA256SUMS`, after downloading all listed assets into one directory.

The `0.2.1` wheel was checked in an isolated Python 3.11 environment on
2026-10-03: dependency consistency, installed imports, tiny NumPy API/CLI
generation, durable-chat restart, ready doctor and two independent portable
deployments. Source and payload hashes were checked before upload. These are
installation checks, not a new large-model benchmark or certification of other
devices. No general throughput or latency improvement is claimed.

## License

The public Python SDK, Reference Compiler, NumPy reference runtime, and public
MLX execution components are licensed under [Apache-2.0](./LICENSE). Installing
this wheel does not require the full development repository.

Optimized native backends remain separately licensed; they are not made Apache
source by this distribution. Model weights, tokenizers, and prepared model
artifacts retain their upstream licenses. See
[Copyright and Licensing](./COPYRIGHT_AND_LICENSING.md) and
[Security](./SECURITY.md).
