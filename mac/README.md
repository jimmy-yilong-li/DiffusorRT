<div align="center">

<img src="../docs/product/assets/diffusorrt-logo.png" alt="DiffusorRT" width="110">

<h1>DiffusorRT for Mac</h1>
<h3>Local conversations. Visible diffusion.</h3>

<p>
<img src="https://img.shields.io/badge/Apple_Silicon-MLX-0891b2?style=flat-square" alt="Apple Silicon with MLX">
<img src="https://img.shields.io/badge/Desktop-0.2.6_preview-7c3aed?style=flat-square" alt="0.2.6 desktop preview, not released">
</p>

<p><a href="#desktop-preview">Desktop preview</a> · <a href="#install-the-sdk">Install</a> ·
<a href="#models">Models</a> · <a href="../windows/README.md">Windows / CUDA</a> ·
<a href="../README.md">All platforms</a></p>

</div>

<p align="center">
<img src="./assets/app-decoding.png" alt="Actual native Mac app showing a reply developing in a masked canvas" width="1100">
<br><sub>The native Mac app candidate, running Qwen3-BD3LM locally through MLX.</sub>
</p>

## Desktop preview

The upcoming **0.2.6 Mac app** makes the diffusion process visible: masked
positions become text, a reply finishes, and the retained steps can be replayed.

- **Live canvas.** Follow block progress, forward passes and remaining masks.
- **Replay.** Inspect retained frames while the app is open.
- **Stop.** Stop at the next decode boundary; a running GPU kernel is not preempted.
- **Conversations.** Reopen saved chats, use multiline prompts and retain an
  uncommitted draft after a stopped or failed reply.
- **Inspector.** See time, visible tokens, context use and MLX memory.
- **Keyboard.** Return sends, Shift-Return adds a line, ⌘N starts a chat and ⌘. stops.

<p align="center">
<img src="./assets/app-reply.png" alt="Completed replies in the Mac app with conversation history and resource inspector" width="1100">
<br><sub>Final replies are saved to history. Intermediate decode frames are not.</sub>
</p>

**Availability:** these are real candidate screenshots, not a downloadable app
release. The app targets macOS 14+ / arm64 and uses an installed Python runtime;
it does not bundle Python or model weights. App distribution terms and
packaging are being finalized. No app download is available yet.

SDK 0.2.6 also prepares an SDK-hosted chat window and structured CLI results.
Neither is included in the currently downloadable 0.2.5 wheel.

## Install the SDK

Available now: **SDK 0.2.5**, Python 3.11 on Apple Silicon.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install "diffusor-rt[compiler,mlx-local] @ https://github.com/jimmy-yilong-li/DiffusorRT/releases/download/v0.2.5/diffusor_rt-0.2.5-py3-none-any.whl"
diffusorrt run "dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1" --prompt "Hello!" --max-tokens 32
```

An existing Python 3.11 conda environment works too. First use downloads about
1.5 GB of weights and creates a roughly 3 GB package; allow about 8.9 GiB free
during preparation. Later runs reuse it. Downloads come from GitHub Releases,
not PyPI.

Continue in the terminal:

```bash
diffusorrt chat "dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1" --session local-demo
```

History is plain, unencrypted SQLite. See the [Python API](../README.md#python-api)
and [CLI guide](../README.md#command-line) for offline use, storage directories,
context management and the local service.

## Models

| Model ID | Current Mac path |
|---|---|
| `dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1` | Recommended first run; MLX fp32, fixed-quota block diffusion, no K/V cache. Infill/edit unsupported. |
| `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1` | Bidirectional MLX fp32, fixed quotas, no K/V cache. Very short outputs and infill quality remain limited. |
| `GSAI-ML/LLaDA-8B-Instruct` | MLX fp32 baseline; about 32 GB for the execution package. A trailing end-of-turn marker is a known limitation. |
| Dream-7B | Exact-D2F requires a separate private native runtime, not included in this download. |

The retained real-model baseline is an Apple M4 Max with 64 GB of memory.
Other Apple hardware is not independently certified here. Model answers can
contain factual or arithmetic errors; the screenshots are not an accuracy or
speed benchmark.

## News

**2026-11-12 — Native desktop preview.** Live masked canvas, replay, cooperative
Stop, persistent conversations and a resource inspector are prepared for the
next Mac release. Public downloads remain at SDK 0.2.5.

**2026-11-02 — SDK 0.2.5 available.**
[Download the library](https://github.com/jimmy-yilong-li/DiffusorRT/releases/tag/v0.2.5)
or read the [project News](../README.md#news).

## License

The public SDK and MLX components use [Apache-2.0](../LICENSE). The native app's
distribution terms are separate and pending. Model weights and tokenizers keep
their upstream licenses.
