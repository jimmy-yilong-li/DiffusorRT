<div align="center">

<img src="./docs/product/assets/diffusorrt-logo.png" alt="DiffusorRT" width="155">

<h1>DiffusorRT</h1>
<h3>Watch language take shape.</h3>
<p>Diffusion-native inference, from Python to your desktop.<br>
An edge-first framework for local, single-request language generation.</p>

<p>
<a href="https://github.com/jimmy-yilong-li/DiffusorRT/releases/tag/v0.2.6"><img src="https://img.shields.io/badge/SDK-0.2.6-6366f1?style=for-the-badge" alt="SDK 0.2.6"></a>
<a href="./mac/README.md"><img src="https://img.shields.io/badge/macOS-Apple_Silicon-0891b2?style=for-the-badge" alt="macOS / Apple Silicon"></a>
<a href="./windows/README.md"><img src="https://img.shields.io/badge/Windows-CUDA_Preview-334155?style=for-the-badge" alt="Windows CUDA development preview"></a>
<a href="./LICENSE"><img src="https://img.shields.io/badge/SDK-Apache_2.0-64748b?style=for-the-badge" alt="Apache-2.0 SDK"></a>
</p>

<p>
<a href="#desktop-experience">Desktop experience</a> ·
<a href="#news">News</a> ·
<a href="#install">Install</a> ·
<a href="#installation-experience">What to expect</a> ·
<a href="#python-api">Python API</a> ·
<a href="#command-line">CLI</a> ·
<a href="./INDEX.md">User guide</a>
</p>

</div>

**One library. Local conversations. Visible diffusion.** Prepare a model,
call it from Python, chat in a window or expose a local HTTP endpoint.
The interfaces share the same model repository and conversation store.

> **SDK 0.2.6 Developer Preview is available.** Mac / MLX has a complete public
> install path. Windows has a portable base SDK and an experimental CUDA
> interface; its native CUDA package is not publicly released yet.

## Desktop experience

<table>
<tr>
<td width="50%" valign="top">
<h3>⌘ Mac · Apple Silicon</h3>
<a href="./mac/README.md"><img src="./mac/assets/page-replay.png" alt="Actual installed Mac SDK: a real reply and recorded masked-sequence replay" width="100%"></a>
<p><strong>Install today.</strong> Local chat, saved conversations, cooperative Stop and recorded decode replay.</p>
<p><sub>Actual installed SDK 0.2.6 on Mac. The image shows a recorded decode replay, not a live performance benchmark.</sub></p>
<p><a href="./mac/README.md"><strong>Mac setup &amp; screenshots →</strong></a></p>
</td>
<td width="50%" valign="top">
<h3>▦ Windows · NVIDIA CUDA</h3>
<a href="./windows/README.md"><img src="./windows/assets/interface-preview.png" alt="Windows interface code rendered with labeled sample data; no CUDA execution" width="100%"></a>
<p><strong>Explore the interface.</strong> Masked-text rendering, model selection, conversation history and sampling controls.</p>
<p><sub>UI preview rendered from the PC frontend with sample data on Mac. Not a Windows machine capture or a GPU run.</sub></p>
<p><a href="./windows/README.md"><strong>Windows / WSL setup &amp; availability →</strong></a></p>
</td>
</tr>
</table>

- **See the decode:** observe a masked canvas becoming text; keep provisional
  frames separate from the final answer.
- **Keep it local:** conversations live on your machine. CLI and GUI reopen
  the same history.
- **Build on the framework:** use Python, CLI, optional TUI or HTTP without
  maintaining a separate model-loading pipeline.
- **Choose the platform deliberately:** one Python SDK, platform-specific
  acceleration. [Do not mix SDK/native release sets](./docs/product/sdk-0.2.6-installation.md).

## News

| Date | Update |
|---|---|
| **2026-11-12** | **0.2.6: local chat and visible diffusion.** Chat window, recorded replay, cooperative Stop, automatic MLX reply sizing and structured CLI timing. [Download](https://github.com/jimmy-yilong-li/DiffusorRT/releases/tag/v0.2.6). |
| **2026-11-12** | **Mac + Windows showcase.** Separate platform guides, interface images and installation walkthroughs. Windows CUDA remains experimental; this SDK includes no CUDA provider. |
| **2026-11-02** | **0.2.5: clearer deployment and upgrades.** Improved invalid-input messages and installed upgrade checks. [Release history](https://github.com/jimmy-yilong-li/DiffusorRT/releases). |

## Install

### Mac — from install to first reply

Use **Python 3.11 on Apple Silicon**. No Git clone or native compiler build is
needed to install the library.

**1. Create an environment and install.**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install "diffusor-rt[compiler,mlx-local,gui] @ https://github.com/jimmy-yilong-li/DiffusorRT/releases/download/v0.2.6/diffusor_rt-0.2.6-py3-none-any.whl"
```

Already using a Python 3.11 conda environment? Run just the `pip install`
line inside it. The package is distributed through GitHub Releases, not PyPI.

**2. Ask a question.**

```bash
diffusorrt run "dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1" --prompt "What is 2+2?" --max-tokens 32
```

The first run downloads and prepares the model automatically. In the recorded
installed check, this prompt returned `2+2 equals 4.`. This example is a
functional check, not an accuracy benchmark.

**3. Open the chat window.**

```bash
diffusorrt gui "dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1"
```

The window opens the model prepared in step 2. On Mac, the `gui` extra supplies
the WebKit window dependencies. Use `--browser` to open the page in your browser.
The separate native Swift App bundle is not part of this release.

### Windows — install the base SDK

Use **Python 3.11** in PowerShell. Calling the environment's Python directly
avoids changing PowerShell's activation-script policy:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install "https://github.com/jimmy-yilong-li/DiffusorRT/releases/download/v0.2.6/diffusor_rt-0.2.6-py3-none-any.whl"
.\.venv\Scripts\python.exe -c "import diffusor_rt; print(diffusor_rt.__version__)"
.\.venv\Scripts\python.exe -m diffusor_rt --help
```

**Expected result: SDK version `0.2.6` and CLI help.** This installs the base
API and package tools, not accelerated model inference. The public Windows
CUDA wheel set is still pending; do not install Mac's `mlx-local` extra or
reuse a Linux provider. [Windows installation details](./windows/README.md#installation-experience).

WSL uses a separate Linux Python environment and Linux native libraries.
Windows DLLs cannot be used in WSL. Linux/A100 has its own acceptance and
release track.

## Installation experience

For the recommended **BD3 0.6B Mac path**:

| Stage | What you see | What happens |
|---|---|---|
| Install the SDK | pip downloads the wheel and requested dependencies | The SDK wheel itself is about **0.91 MB**; extras are separate and considerably larger. No model is loaded. |
| First model run | Download/preparation messages before the reply | About **1.5 GB** of source weights are downloaded and a roughly **3 GB** execution package is prepared. |
| Prepare disk space | Keep about **8.9 GiB free** for model preparation | This includes temporary-build headroom, not just the final cache. Python dependencies need additional space. |
| Run again | The prepared package is reused | The first-run download/conversion is not repeated. A new process still needs to load the model into memory. |
| Open chat | The prepared model and saved conversations appear | Type normally, watch the decode, stop a supported request or revisit history. |

Download and preparation time depend on your connection and hardware. We do
not promise a universal installation time or model speed.

<details>
<summary><strong>Check the installation and handle common first-run surprises</strong></summary>

```bash
python -c "import diffusor_rt; print(diffusor_rt.__version__)"
python -m diffusor_rt --help
diffusorrt models "dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1" --json
```

These commands do not load model weights.

- **`doctor` says “not ready” before the first run:** prepare with `run` or
  `pull`. Exit status 1 here does not mean the library installation failed.
- **First launch takes longer:** it downloads and converts the model. Later
  launches reuse the prepared package.
- **No prepared model in the window:** run the step-2 command or `pull` first.
- **Missing an extra:** rerun the versioned wheel install command with the needed
  extras. The package is not currently installed by a plain PyPI-name command.
- **Offline use:** prepare online once, then add `--offline` to model commands.
  `doctor --offline` is metadata-only, not an inference-readiness check.
- **Low disk space:** source downloads and prepared packages both occupy space;
  the 8.9 GiB preparation allowance is not a RAM requirement.

</details>

<details>
<summary><strong>Upgrade without deleting your models or conversations</strong></summary>

Use the same release URL with `pip install --upgrade`. The checked
0.2.5-to-0.2.6 path preserved the prepared package and existing conversation.

Model-ID loading rebuilds stale execution packages when necessary. An explicit
package path remains pinned to that package. Keep your downloads and history;
do not erase them to resolve a version mismatch.

Older session identities can differ after model/configuration changes. Start a
new `--session` to preserve the old history, and delete a session only when you
intend to discard it. [Version pairing and checksums](./docs/product/sdk-0.2.6-installation.md).

</details>

## Python API

```python
from diffusor_rt import Engine

with Engine.from_pretrained(
    "dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1",
    backend="mlx_local",
) as engine:
    result = engine.generate("Hello!", max_tokens=32)
    print(result.text)
    print(result.elapsed_seconds)
```

Use `engine.chat_result(message)` for a structured conversation reply, or
`engine.chat(message)` for a string. The context manager closes the engine
and releases its resources. Model IDs resolve automatically; a prepared local
package path also works.

Try the standalone [quickstart script](./examples/quickstart.py) after installing
the wheel. [User index](./INDEX.md) · [Mac API examples](./mac/README.md#python-and-command-line).

## Command line

```bash
MODEL="dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1"

diffusorrt run "$MODEL" --prompt "Hello!"
diffusorrt chat "$MODEL" --session local-demo
diffusorrt run "$MODEL" --prompt-file ./prompt.txt
printf 'Hello!\n' | diffusorrt run "$MODEL" --prompt-file -
diffusorrt models "$MODEL" --json
```

In interactive chat, `/help` lists commands, `/new` opens a new conversation,
and `/multiline` starts a draft; finish with `/send` or discard with `/cancel`.
Use `--cache-root PATH` and `--data-root PATH` to choose local storage.

MLX replies run to model termination with a **1,024-token safety ceiling**.
Set `--max-tokens` for a smaller budget. GUI Stop takes effect at a supported
decode boundary, not in the middle of a GPU kernel.

### Model manager and local service

The optional TUI and HTTP adapter use the same library. Add their extras:

```bash
python -m pip install "diffusor-rt[compiler,mlx-local,gui,tui,serve] @ https://github.com/jimmy-yilong-li/DiffusorRT/releases/download/v0.2.6/diffusor_rt-0.2.6-py3-none-any.whl"
diffusorrt models
diffusorrt serve "$MODEL" --host 127.0.0.1 --port 8000
```

`diffusorrt models` without an ID opens the optional model manager in a terminal.
Run the HTTP service in its own terminal. The local server is not a hardened
public multi-user serving endpoint.

## Supported models and platforms

| Model | Public Mac path | Notes |
|---|---|---|
| **Qwen3-BD3LM 0.6B** | Recommended first run · MLX fp32 | Generate/chat; no private runtime needed. No infill/edit. |
| **Qwen3-MDLM 0.6B** | MLX fp32 baseline | Generate/chat; short-answer and infill quality remain limited. |
| **LLaDA-8B Instruct** | Existing MLX fp32 baseline | Much larger: about 32 GB prepared weights. A trailing `<\|eot_id\|>` may remain visible. |
| **Dream-7B** | Separate native runtime required | Its exact-D2F profile is not a standalone public-wheel path. |

The 0.2.6 installed walkthrough covered the two 0.6B models, CLI/API, persistent
chat, HTTP, upgrade and the Mac page/replay. It did not repeat a large-model run,
interactive TUI acceptance or Windows/CUDA hardware validation.

Small models can give wrong facts or arithmetic. A complete response is not an
accuracy guarantee. Mac sampling remains greedy; nonzero temperature is rejected.
The Windows CUDA sampling work has separate device evidence and distribution.

Conversations are stored in **plain, unencrypted local SQLite**. Summaries are
opt-in; applications can provide a `MemoryProvider`. Automatic long-term memory
collection is not enabled by default.

## Downloads and guides

- [SDK 0.2.6 release](https://github.com/jimmy-yilong-li/DiffusorRT/releases/tag/v0.2.6)
- [Mac installation and screenshots](./mac/README.md)
- [Windows / WSL installation and interface preview](./windows/README.md)
- [Matching SDK/native installations](./docs/product/sdk-0.2.6-installation.md)
- [User guide index](./INDEX.md) · [Report an issue](https://github.com/jimmy-yilong-li/DiffusorRT/issues)

The public download is `diffusor_rt-0.2.6-py3-none-any.whl`:

```text
SHA-256 86c9cee779bf65e127e3d5326caed806cf7bdada0d080ae41a5d540347116acb
```

The wheel contains the Python modules and resources needed to run the SDK.
The repository's selected API excerpts are reading examples, not a complete
editable installation. No native Mac App bundle, CUDA provider or model weights
are included. Earlier 0.2.0–0.2.4 downloads are retired; 0.2.5 remains the
previous SDK-only rollback reference.

## Citation and license

For research, cite the software version through [CITATION.cff](./CITATION.cff)
or GitHub's **Cite this repository** entry.

The SDK, Reference Compiler, NumPy reference and public MLX components are
[Apache-2.0](./LICENSE). Native backends and model weights retain their own
terms. The separate Swift App's distribution license remains unresolved.
[Licensing](./COPYRIGHT_AND_LICENSING.md) · [Security](./SECURITY.md).
