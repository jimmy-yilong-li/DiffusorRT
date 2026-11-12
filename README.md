<div align="center">

<img src="./docs/product/assets/diffusorrt-logo.png" alt="DiffusorRT" width="190">

<h1>DiffusorRT</h1>

<h3>Diffusion language models. Local hardware. One runtime.</h3>

<p>From a masked canvas to a local conversation.</p>

<p>
<a href="https://github.com/jimmy-yilong-li/DiffusorRT/releases/tag/v0.2.6"><img src="https://img.shields.io/badge/Public_SDK-0.2.6-2563eb?style=flat-square" alt="SDK 0.2.6 Developer Preview"></a>
<a href="#chat-window"><img src="https://img.shields.io/badge/Live_decode-GUI-7c3aed?style=flat-square" alt="Local chat and decode replay"></a>
<a href="./mac/README.md"><img src="https://img.shields.io/badge/Mac-Apple_Silicon-0891b2?style=flat-square" alt="Mac / Apple Silicon"></a>
<a href="./windows/README.md"><img src="https://img.shields.io/badge/Windows-CUDA_development-64748b?style=flat-square" alt="Windows / CUDA in development"></a>
</p>

<p>
<a href="#news">News</a> ·
<a href="./INDEX.md">User guide</a> ·
<a href="#install">Install</a> ·
<a href="#python-api">Python API</a> ·
<a href="#command-line">Command line</a> ·
<a href="#chat-window">Chat window</a> ·
<a href="#model-manager-and-local-service">Model manager</a> ·
<a href="#supported-models-and-platforms">Supported models</a> ·
<a href="#citation">Citation</a> ·
<a href="#license">License</a>
</p>

<p><a href="./mac/README.md">Mac</a> &nbsp; / &nbsp;
<a href="./windows/README.md">Windows &amp; WSL</a> &nbsp; / &nbsp;
<a href="https://github.com/jimmy-yilong-li/DiffusorRT/releases">Downloads</a></p>

</div>

> [!NOTE]
> **SDK 0.2.6 Developer Preview.** Install the library, then open a local chat
> with `diffusorrt gui`. This release contains the Python wheel, not a native
> Mac App bundle or a CUDA provider. See the [installation and compatibility guide](./docs/product/sdk-0.2.6-installation.md).

<p align="center">
<a href="./mac/README.md"><img src="./mac/assets/page-replay.png" alt="Installed SDK 0.2.6 chat page with a real masked-sequence replay" width="1000"></a>
<br><sub>Actual installed SDK 0.2.6 · Qwen3-BD3LM 0.6B · browser view of recorded decode frames, separate from the final answer.</sub>
</p>

DiffusorRT provides model loading, generation, conversation history, context
management, and a local HTTP service through one importable library. Models use
the same loader and runtime; the command line and optional TUI call that library.

<table>
<tr>
<td width="50%"><h3>◈ Watch diffusion happen</h3>
See masked positions resolve into text, then replay the recorded decode.
Preview frames stay separate from the final reply.</td>
<td width="50%"><h3>⌘ One library, your interface</h3>
Use Python, the command line, a local chat window or HTTP.
The interfaces share model preparation and conversations.</td>
</tr>
<tr>
<td width="50%"><h3>↔ Local by design</h3>
Run on your own hardware. Keep conversations locally and extend the
framework through its APIs.</td>
<td width="50%"><h3>▦ Edge-first execution</h3>
Shared diffusion policies, explicit resource ownership and checked shutdown.
Focused on single-request inference, not multi-tenant serving.</td>
</tr>
</table>

**0.2.6 is a Developer Preview.** Its delivery is an installable Apache-2.0
Python wheel, including the local chat window, not a new source-code export.
Previously published examples and API excerpts remain available for reference.
Optimized native Apple/CUDA backends and
model weights are separate and are not included in this wheel.

## News

| Date | Update |
|---|---|
| **2026-11-12** | **SDK 0.2.6 — local chat, visible diffusion.** Chat window, recorded decode replay, cooperative Stop, automatic MLX reply sizing and structured CLI timing. [Download](https://github.com/jimmy-yilong-li/DiffusorRT/releases/tag/v0.2.6) · [Mac screenshots](./mac/README.md). |
| **2026-11-12** | **Separate platform tracks.** Reviewed CUDA PC source adds seeded sampling and retry repairs; native CUDA downloads remain separate and unavailable in this SDK release. [Windows / WSL status](./windows/README.md) · [Do not mix release sets](./docs/product/sdk-0.2.6-installation.md). |
| **2026-11-02** | **SDK 0.2.5.** Clearer deployment input errors and verified installed upgrades. [Release history](https://github.com/jimmy-yilong-li/DiffusorRT/releases). |

## Install

Use **Python 3.11**. The accelerated example below needs **Apple Silicon and
macOS**. It uses Qwen3-BD3LM 0.6B and does not require a private backend wheel.

Install directly from the GitHub release; no Git clone or compiler build is
needed to install the Python library:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install "diffusor-rt[compiler,mlx-local] @ https://github.com/jimmy-yilong-li/DiffusorRT/releases/download/v0.2.6/diffusor_rt-0.2.6-py3-none-any.whl"
```

Get your first reply:

```bash
diffusorrt run "dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1" --prompt "Hello!"
```

This first model run downloads about 1.5 GB of weights and prepares a roughly
3 GB execution package. Allow about 8.9 GiB of free disk space for preparation.
Later runs reuse the cache. Import and help commands do not load a model.

An existing Python 3.11 conda environment works too; skip the first two commands
when it is already active. The `compiler` extra installs the dependencies for
downloading and preparing models, and `mlx-local` installs the Apple backend.

Alternatively, download the wheel from [Release assets](https://github.com/jimmy-yilong-li/DiffusorRT/releases/tag/v0.2.6)
and install it locally:

```bash
python -m pip install "./diffusor_rt-0.2.6-py3-none-any.whl[compiler,mlx-local]"
```

For only the base API, package tools, and NumPy reference runtime, omit the
extras. This does not install an accelerated backend or model-download tools:

```bash
python -m pip install "https://github.com/jimmy-yilong-li/DiffusorRT/releases/download/v0.2.6/diffusor_rt-0.2.6-py3-none-any.whl"
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
`pull` to obtain corrected metadata. Version `0.2.0` has known BD3 early-stop
and chat-metadata defects; use the current release. Do not mix SDK and native
provider downloads from different release sets.

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

Upgrading from `0.2.4` corrects `deploy` input errors. `deploy` expects a prepared
package directory, not a model ID; invalid targets now explain which command
to use, and `--once --json` returns a structured error. This patch does not
change decoding, model-package identities or conversation storage.

Upgrading from `0.2.5` adds the local chat window (`diffusorrt gui`),
cooperative Stop on the cache-free MLX profiles, and the
finish reason and elapsed time in `run --json`. Replies no longer stop at 128
tokens by default: they run until the model ends them, up to 1,024 tokens. Model-package identities and conversation
storage are unchanged: prepared models and saved conversations are reused.

The older `0.2.1` wheel retains the long-output scheduling and allocator-cache
release defects corrected in `0.2.2`; use the current download. Output caps can
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
    result = engine.generate("Hello!")
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
    print(engine.generate("Hello!").text)
```

A prepared package is a DiffusorRT execution package, not an arbitrary directory
of Hugging Face weights. `offline=True` can also use previously cached source.

The same API is available as a ready-to-run script:

```bash
# From this repository, after installing the wheel:
python examples/quickstart.py --prompt "Hello!"
```

Download [quickstart.py](./examples/quickstart.py) separately if you do not want
to clone the repository. It is a thin API example, not a second runtime.

## Command line

```bash
MODEL="dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1"

# First run: download, prepare, and reply. Later runs reuse the package.
diffusorrt run "$MODEL" --prompt "Hello!"

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
printf 'Hello!\n' | diffusorrt run "$MODEL" --prompt-file -
```

`run` can prepare an uncached model itself; `pull` is optional. Use `--offline`
to forbid network access after caching. Inspect each command with `--help` for
supported runtime parameters.

Before `pull`, `doctor` exits with status 1 when the model is not prepared;
follow its preparation instruction. This does not mean installation failed.
`chat` reuses the prepared model and resumes the named conversation on later
invocations.

A reply runs until the model ends it; there is no output limit to set. On Apple
Silicon a reply that has not ended after 1,024 tokens stops there and is marked
as cut short: a small model that keeps going is usually repeating itself.
`--max-tokens` (and `max_tokens=` in Python) still sets a lower limit when you
want one. A complete reply is not a quality guarantee.

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

## Chat window

On an Apple Silicon Mac, `diffusorrt gui` opens a local chat over the same
library and conversations as `chat`, and shows the decode while a reply is
written. [DiffusorRT for Mac](./mac/README.md) has installation steps, screenshots
and Mac News. A separate native Swift app is still a development preview and
is not a downloadable asset in this release.

<p align="center">
<img src="./mac/assets/page-reply.png" alt="Actual installed SDK 0.2.6 chat window with a completed reply and decode replay controls" width="1000">
<br><sub>The window delivered by this SDK, running locally on Apple Silicon.</sub>
</p>

The chat page for other platforms is in the wheel as well; it is not part of
what this release verifies (see the platform table below).

## Model manager and local service

Install the optional TUI and HTTP service dependencies:

```bash
python -m pip install "diffusor-rt[compiler,mlx-local,serve,tui] @ https://github.com/jimmy-yilong-li/DiffusorRT/releases/download/v0.2.6/diffusor_rt-0.2.6-py3-none-any.whl"
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
| `GSAI-ML/LLaDA-8B-Instruct` | Apple/MLX fp32 baseline. Much larger: the execution package is about 32 GB. Its replies end with a visible `<|eot_id|>` marker; this is a known defect. |
| `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1` | Bidirectional Apple/MLX fp32, shared fixed quotas with 64-token blocks, no K/V cache. Selected user examples improve; short outputs and infill/edit quality remain limited. Renewed certification of the changed default is pending. |
| Dream-7B | Its exact-D2F profile needs a compatible private native runtime, which is not included in this release. |

The retained real-model baseline for the two Apache-only profiles names
**Apple M4 Max / 64 GB / MLX fp32**. This SDK patch does not replace that evidence
with a new fixed-source acceptance for every model. Other Apple hardware is not
independently certified by this release. NumPy remains a portable reference,
not a claim of practical 8B CPU
performance. CUDA, reduced-precision BD3, BD3 caching, and mobile acceleration
are not provided by this public wheel.

### Platform and artifact availability

The SDK wheel is pure Python and can be shared across platforms. Accelerated
execution still needs platform-compatible dependencies: MLX on Apple Silicon,
or a separately supplied CUDA native wheel on its advertised systems. Windows
native binaries and Linux/WSL binaries are different artifacts. Each platform
has its own installation and release checks; the SDK version alone does not
establish support. Native-runtime versions need not match the SDK version.

GUI availability and accelerated inference support are separate. SDK **0.2.6**
ships the Mac page-backed chat window; the separate native app is not included.

| Platform | 0.2.6 delivery scope | Remaining platform work |
|---|---|---|
| Apple Silicon / macOS | MLX: Python, CLI, conversations, local HTTP, and the [chat window](./mac/README.md). | Separate native App licensing and distribution. |
| [Windows / RTX 4070](./windows/README.md) | No bundled CUDA provider. | Experimental BF16 chat window; a separate Windows native wheel and its own release acceptance are required. |
| [WSL2 / NVIDIA](./windows/README.md#windows-and-wsl-are-different-installations) | No bundled CUDA provider. | Experimental Linux CUDA execution; WSL uses Linux artifacts, not Windows DLLs. |
| Linux / A100 | No bundled CUDA provider. | Separate CUDA server candidate and hardware acceptance; not covered by the Mac preview. |
| Portable Python / NumPy | Base API, package tools and reference execution. | No new accelerated backend. |

Windows GUI results are collaborator-reported on one RTX 4070, not general
Windows qualification; INT8 was not admitted for that model. Native CUDA
artifacts and GUI installation will have their own explicit availability and
platform requirements. NumPy is a correctness reference, not a practical
large-model CPU performance promise.

The library exposes generation, conversation, infill, and edit interfaces;
availability depends on the selected model's decoding contract. In particular,
BD3 does not support infill or edit. Unsupported combinations fail explicitly.
Positive-budget generation or conversation requests that produce no visible
text raise `EmptyGenerationError` without committing a turn or silently retrying.

## Release and verification

This update adds no product source modules. Earlier releases provided these
reference excerpts and example:

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

The 0.2.6 download is the Python wheel:

| Asset | SHA-256 |
|---|---|
| `diffusor_rt-0.2.6-py3-none-any.whl` | `86c9cee779bf65e127e3d5326caed806cf7bdada0d080ae41a5d540347116acb` |

Compare them with `shasum -a 256 <file>` on macOS or `sha256sum <file>` on
Linux.

The rebuilt wheel is checked in a fresh environment on Apple Silicon, outside
the source tree: both 0.6B models, CLI JSON/stdin, model inspection, cached
preparation, cross-process chat, and a real GUI reply and decode replay.
The screenshots above come from this installed wheel. Earlier candidate checks
also covered LLaDA and reference forward comparisons; those are not fresh
measurements of this rebuilt artifact. The interactive TUI and CUDA hardware
were not exercised. Existing model mistakes and the BD3 difference from upstream
padded sampling remain open; SDK usability does not establish answer accuracy,
renewed certification or a general throughput/latency improvement.

## Citation

For research that uses DiffusorRT, use the repository's **Cite this repository**
entry or [CITATION.cff](./CITATION.cff). Cite the software version used in your
experiments for reproducibility.

## License

The public Python SDK, Reference Compiler, NumPy reference runtime, and public
MLX execution components are licensed under [Apache-2.0](./LICENSE). Installing
this wheel does not require the full development repository.

The native Mac app's distribution license is still being finalized. The SDK's
Apache-2.0 notice does not settle that separate artifact's terms.

Optimized native backends remain separately licensed; they are not made Apache
source by this distribution. Model weights, tokenizers, and prepared model
artifacts retain their upstream licenses. See
[Copyright and Licensing](./COPYRIGHT_AND_LICENSING.md) and
[Security](./SECURITY.md).

## Earlier updates

<details>
<summary>Development milestones and earlier releases</summary>

**2026-11-10 — Desktop GUI development milestone (historical).**
A shared local chat interface is in development over the existing model and
conversation APIs: model selection, history reopening, code rendering and a
collapsible Resources panel. The Windows/RTX 4070 prototype has collaborator
evidence; Mac window, input and unified-memory integration are being prepared.
At this milestone the downloadable SDK was **0.2.5**. SDK 0.2.6 now delivers
the verified local chat window shown above; Windows/WSL2 and A100 CUDA
components remain separate from the public SDK distribution.

**2026-11-02 — SDK 0.2.5 deployment error handling.**
[Download the wheel](https://github.com/jimmy-yilong-li/DiffusorRT/releases/tag/v0.2.5).
`deploy` now distinguishes a model ID, missing directory or ordinary file from
a prepared execution package, explains how to use `run` or `pull`, and preserves
structured JSON errors. SDK-versioned validation metadata now covers 0.2.3–0.2.5.
This is a maintenance preview, not a model-quality or performance update.
Only the wheel is newly distributed; no backend binaries, weights or new source
modules accompany it. Existing releases remain unchanged.

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

</details>
