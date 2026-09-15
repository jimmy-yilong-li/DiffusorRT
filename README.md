<div align="center">

<img src="./docs/product/assets/diffusorrt-logo.png" alt="DiffusorRT" width="190">

<h1>DiffusorRT</h1>

<p><strong>A Python runtime for diffusion language models on local hardware.</strong></p>

<p>
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

**0.2.0 is a Developer Preview, released in stages.** This first batch provides
an installable Apache-2.0 Python wheel and a small selection of API source files,
not the full development repository. Optimized native Apple/CUDA backends and
model weights are separate and are not included in this wheel.

## Install

Use **Python 3.11**. The accelerated example below needs **Apple Silicon and
macOS**. It uses Qwen3-BD3LM 0.6B and does not require a private backend wheel.

Install directly from the GitHub release; no Git clone or compiler build is
needed to install the Python library:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install "diffusor-rt[compiler,mlx-local] @ https://github.com/jimmy-yilong-li/DiffusorRT/releases/download/v0.2.0/diffusor_rt-0.2.0-py3-none-any.whl"
```

An existing Python 3.11 conda environment works too; skip the first two commands
when it is already active. The `compiler` extra installs the dependencies for
downloading and preparing models, and `mlx-local` installs the Apple backend.

Alternatively, download the wheel from [Release assets](https://github.com/jimmy-yilong-li/DiffusorRT/releases/tag/v0.2.0)
and install it locally:

```bash
python -m pip install "./diffusor_rt-0.2.0-py3-none-any.whl[compiler,mlx-local]"
```

For only the base API, package tools, and NumPy reference runtime, omit the
extras. This does not install an accelerated backend or model-download tools:

```bash
python -m pip install "https://github.com/jimmy-yilong-li/DiffusorRT/releases/download/v0.2.0/diffusor_rt-0.2.0-py3-none-any.whl"
```

The package name is `diffusor-rt`; the Python import is `diffusor_rt`.
This preview is distributed through GitHub Releases, not PyPI. The
`py3-none-any` tag describes the Python wheel, not GPU or model support on every
platform.

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
        "Explain diffusion language models briefly.",
        max_tokens=16,
    )
    print(result.text)
```

The first run downloads about **1.5 GB** of weights and builds a roughly
**3 GB** execution package. Allow additional room for dependencies and temporary
build files. Later runs reuse the package. Model preparation and generation are
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
python examples/quickstart.py --prompt "Explain dLLMs briefly." --max-tokens 16
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

diffusorrt run "$MODEL" --prompt "Explain dLLMs briefly." --max-tokens 16

# Start or resume a conversation with local history.
diffusorrt chat "$MODEL" --session local-demo --max-tokens 16
```

`run` can prepare an uncached model itself; `pull` is optional. Use `--offline`
to forbid network access after caching. Inspect each command with `--help` for
supported runtime parameters.

Conversation history is stored locally in **plain, unencrypted SQLite**.
Context management keeps requests within the model's token budget. Summaries
are opt-in and require additional model work. DiffusorRT is a framework, not a
fixed chatbot: applications can provide their own `MemoryProvider`; there is no
automatic long-term memory collection or retrieval by default.

## Model manager and local service

Install the optional TUI and HTTP service dependencies:

```bash
python -m pip install "diffusor-rt[compiler,mlx-local,serve,tui] @ https://github.com/jimmy-yilong-li/DiffusorRT/releases/download/v0.2.0/diffusor_rt-0.2.0-py3-none-any.whl"
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

Real-model acceptance for the two Apache-only profiles names **Apple M4 Max /
64 GB / MLX fp32**. Other Apple hardware is not independently certified by this
release. NumPy remains a portable reference, not a claim of practical 8B CPU
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

The [0.2.0 release](https://github.com/jimmy-yilong-li/DiffusorRT/releases/tag/v0.2.0)
contains the wheel, standalone example, and `SHA256SUMS`. Downloaded files can be
checked on macOS with `shasum -a 256 -c SHA256SUMS`, or on Linux with
`sha256sum -c SHA256SUMS`, after downloading all listed assets into one directory.

The wheel was checked in an isolated Python 3.11 environment on 2026-09-15:
dependencies, imports, CPU fixture generation through API/CLI/example, durable
chat, HTTP health/generation/shutdown, and the TUI profile dialog. These are
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
