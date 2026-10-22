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

**0.2.4 is a Developer Preview, released in stages.** This batch provides
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
python -m pip install "diffusor-rt[compiler,mlx-local] @ https://github.com/jimmy-yilong-li/DiffusorRT/releases/download/v0.2.4/diffusor_rt-0.2.4-py3-none-any.whl"
```

Get your first reply:

```bash
diffusorrt run "dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1" --prompt "Hello!" --max-tokens 32
```

This first model run downloads about 1.5 GB of weights and prepares a roughly
3 GB execution package. Allow about 8.9 GiB of free disk space for preparation.
Later runs reuse the cache. Import and help commands do not load a model.

An existing Python 3.11 conda environment works too; skip the first two commands
when it is already active. The `compiler` extra installs the dependencies for
downloading and preparing models, and `mlx-local` installs the Apple backend.

Alternatively, download the wheel from [Release assets](https://github.com/jimmy-yilong-li/DiffusorRT/releases/tag/v0.2.4)
and install it locally:

```bash
python -m pip install "./diffusor_rt-0.2.4-py3-none-any.whl[compiler,mlx-local]"
```

For only the base API, package tools, and NumPy reference runtime, omit the
extras. This does not install an accelerated backend or model-download tools:

```bash
python -m pip install "https://github.com/jimmy-yilong-li/DiffusorRT/releases/download/v0.2.4/diffusor_rt-0.2.4-py3-none-any.whl"
```

The package name is `diffusor-rt`; the Python import is `diffusor_rt`.
This preview is distributed through GitHub Releases, not PyPI. The
`py3-none-any` tag describes the Python wheel, not GPU or model support on every
platform.

<details>
<summary>Upgrading an existing installation</summary>

To upgrade an existing installation, use the same install command with
`--upgrade`. Model-ID loading rebuilds stale compiled packages automatically;
you do not need to erase model downloads or conversation history. Explicit
package paths stay pinned to their contents, so prepare a new package with
`pull` to obtain corrected metadata. The older `0.2.0` assets remain available,
but that version has the BD3 early-stop and chat-metadata defects fixed here.

An empty session left by a failed first chat automatically rebinds after a
package rebuild when its system prompt is unchanged. Sessions containing
messages preserve their history and refuse a changed execution identity.
Choose a new `--session` name to keep the old history; inspect it with
`diffusorrt chat --list-sessions`, and delete it only when you intend to discard
it. In particular, MDLM's changed default can require a new session.

Upgrading from `0.2.2` also changes MDLM's decode schedule. Applications that
previously accepted an empty `generate()` result should now catch
`diffusor_rt.EmptyGenerationError`; raw `max_tokens=0` generation remains valid.
The runtime does not silently retry or invent a response.

Upgrading from `0.2.3` adds model descriptions, structured chat replies and
terminal input commands. Existing `Engine.chat()` calls still return strings.

The older `0.2.1` wheel retains the long-output scheduling and allocator-cache
release defects corrected in `0.2.2`; its assets are unchanged. Output caps can
still truncate an answer, and partial final blocks can legitimately change
decoding. The small default checkpoint can give incorrect facts or arithmetic.
Successful generation is not a correctness guarantee for its answers.

</details>

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
    print(result.elapsed_seconds)
```

Use `engine.chat_result(message)` for a structured chat reply with `.text`,
`.finish_reason`, `.elapsed_seconds` and `.to_dict()`. `engine.chat(message)`
keeps its string return value. For preparation progress, pass
`on_stage=print` to `Engine.from_pretrained()`.

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

# First run: download, prepare, and reply. Later runs reuse the package.
diffusorrt run "$MODEL" --prompt "Hello!" --max-tokens 32

# Keep a conversation and resume it later with the same session name.
diffusorrt chat "$MODEL" --session local-demo

# Optional preparation and diagnostics; no separate pull is required above.
diffusorrt pull "$MODEL"
diffusorrt doctor "$MODEL"

# Inspect model operations without downloading weights or installing the TUI.
diffusorrt models "$MODEL" --json
diffusorrt models --list

# Read one prompt from a UTF-8 file, or pipe it through standard input.
diffusorrt run "$MODEL" --prompt-file ./prompt.txt
printf 'Hello!\n' | diffusorrt run "$MODEL" --prompt-file - --max-tokens 32
```

`run` can prepare an uncached model itself; `pull` is optional. Use `--offline`
to forbid network access after caching. Inspect each command with `--help` for
supported runtime parameters.

Before `pull`, `doctor` exits with status 1 when the model is not prepared;
follow its preparation instruction. This does not mean installation failed.
`chat` reuses the prepared model and resumes the named conversation on later
invocations. A generation token limit can still truncate a long answer; it is
not a quality guarantee.

In an interactive `run` or `chat`, `/help` lists commands, `/new` starts another
conversation while retaining the old history, and `/multiline` opens a draft.
Finish the draft with `/send` or discard it with `/cancel`. Prefix a literal
slash message with `//`. The generation-start message reports that a request is
running; replies arrive as complete text.

All model commands accept `--cache-root PATH` and `--data-root PATH` to select
the compiled-package cache and local configuration/history directories.

Conversation history is stored locally in **plain, unencrypted SQLite**.
Context management keeps requests within the model's token budget. Summaries
are opt-in and require additional model work. DiffusorRT is a framework, not a
fixed chatbot: applications can provide their own `MemoryProvider`; there is no
automatic long-term memory collection or retrieval by default.

## Model manager and local service

Install the optional TUI and HTTP service dependencies:

```bash
python -m pip install "diffusor-rt[compiler,mlx-local,serve,tui] @ https://github.com/jimmy-yilong-li/DiffusorRT/releases/download/v0.2.4/diffusor_rt-0.2.4-py3-none-any.whl"
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
| `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1` | Bidirectional Apple/MLX fp32, shared fixed quotas with 64-token blocks, no K/V cache. Selected user examples improve; short outputs and infill/edit quality remain limited. Renewed certification of the changed default is pending. |
| Dream-7B | Its exact-D2F profile needs a compatible private native runtime, which is not included in this release. |

The retained real-model baseline for the two Apache-only profiles names
**Apple M4 Max / 64 GB / MLX fp32**. This SDK patch does not replace that evidence
with a new fixed-source acceptance for every model. Other Apple hardware is not
independently certified by this release. NumPy remains a portable reference,
not a claim of practical 8B CPU
performance. CUDA, reduced-precision BD3, BD3 caching, and mobile acceleration
are not provided by this public wheel.

The library exposes generation, conversation, infill, and edit interfaces;
availability depends on the selected model's decoding contract. In particular,
BD3 does not support infill or edit. Unsupported combinations fail explicitly.
Positive-budget generation or conversation requests that produce no visible
text raise `EmptyGenerationError` without committing a turn or silently retrying.

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

The [0.2.4 release](https://github.com/jimmy-yilong-li/DiffusorRT/releases/tag/v0.2.4)
contains the wheel, standalone example, and `SHA256SUMS`. Downloaded files can be
checked on macOS with `shasum -a 256 -c SHA256SUMS`, or on Linux with
`sha256sum -c SHA256SUMS`, after downloading all listed assets into one directory.

Earlier fixed candidates were exercised on BD3/MDLM 0.6B through CLI/API,
Local deployment, saved profiles, TUI, HTTP, conversation restart and close.
The 0.2.4 artifact has separate installation, API/CLI and download checks using
a tiny NumPy package. No new real-model run accompanies this release. Existing
model mistakes and the BD3 difference from upstream padded sampling remain
open; SDK usability does not establish answer accuracy, renewed certification
or a general throughput/latency improvement.

## License

The public Python SDK, Reference Compiler, NumPy reference runtime, and public
MLX execution components are licensed under [Apache-2.0](./LICENSE). Installing
this wheel does not require the full development repository.

Optimized native backends remain separately licensed; they are not made Apache
source by this distribution. Model weights, tokenizers, and prepared model
artifacts retain their upstream licenses. See
[Copyright and Licensing](./COPYRIGHT_AND_LICENSING.md) and
[Security](./SECURITY.md).

## News

**2026-10-21 — SDK 0.2.4 client and runtime maintenance.**
[Download the preview](https://github.com/jimmy-yilong-li/DiffusorRT/releases/tag/v0.2.4).
Inspect model operations without downloading weights, configure shared storage
directories, and submit file/stdin prompts or multiline chat drafts. `/help`
and `/new` are available in interactive chat. Python adds `chat_result()`,
result timing/serialization, public error imports and preparation callbacks;
existing `chat()` calls keep returning strings. Missing extras name a complete
release-wheel installation command.

Shared runtime fixes preserve fixed infill/edit text, return a continuous
prefix for incomplete output, keep the original error when cleanup also fails,
and correct request/phase timing. Output projection and confidence buffers use
bounded temporary storage. Model accuracy, general speedup, streamed replies,
running GPU cancellation and private native binaries remain outside this update.

**2026-10-20 — SDK 0.2.3 user-path and decoder repairs.**
[Download the preview](https://github.com/jimmy-yilong-li/DiffusorRT/releases/tag/v0.2.3).
Local catalogs now use the same deployment policy in `run`, `doctor`, `deploy`
and `serve`; saved Local profiles reopen their own catalog. Missing TUI
dependencies produce an installation hint instead of a traceback.

Qwen3-MDLM now uses shared bidirectional fixed-quota decoding instead of a
borrowed threshold schedule. Installed arithmetic, translation and story
examples improved; the correct schedule can require more forwards and take
longer. This changes outputs and is **not a general speedup or accuracy claim**.
The small checkpoints still make factual, translation and recall errors;
MDLM's very short outputs and infill/edit quality remain limited.

Positive-budget `Engine.generate()` now raises `EmptyGenerationError` when
there is no visible text; `.result` retains the generated result and requested
diagnostics. CLI and HTTP report the same failure without publishing an empty
conversation turn. HTTP response waiting is configurable, but does not cancel
already-running GPU work. Memory reporting now measures MLX requests selected
through `backend="auto"`. This SDK-only release adds no native binary or source
category and does not renew every model's certification.

**2026-10-07 — SDK 0.2.2 maintenance update.**
[Download the preview](https://github.com/jimmy-yilong-li/DiffusorRT/releases/tag/v0.2.2).
BD3 now uses the corrected per-block round budget and forwards only through
the current block. Successful close waits for device work and releases MLX
allocator-cache buffers; failed cleanup remains visible and retryable. BD3
`infill()` and `edit()` now clearly report that its decoding policy does not
support those operations, rather than exposing an internal strategy name.
Empty positive-budget chat replies are not committed, and HTTP reports an
`empty_generation` error. Empty failed sessions can recover after a package
rebuild; intact stale packages are reported as `needs_rebuild`.

Qwen3-MDLM now defaults to its bidirectional baseline instead of a block-causal
D2F overlay. It no longer needs the private native K/V wheel on that path,
but this changed default has not received renewed model certification.
Optional evaluation tools also correct numeric-answer boundaries and sample
unique question IDs. These repairs can change outputs relative to `0.2.1`;
they do not establish better dataset accuracy or a general speedup. The
release remains SDK-only: no private native binary or wider source export.

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
