# SDK 0.2.6 — matching installations

One Python SDK does not mean one interchangeable GPU installation. Use a
separate Python 3.11 environment for each platform and its native providers.

| Environment | Public 0.2.6 installation | Native distribution |
|---|---|---|
| Apple Silicon / macOS | SDK 0.2.6 + `compiler,mlx-local,gui` extras; MLX 0.31.x | No private runtime needed for the two 0.6B baseline models. |
| Native Windows / CUDA PC | Base SDK and API are portable; this is not a complete accelerated install | No public matched CUDA wheel set in this release. |
| WSL2 / Linux CUDA PC | Linux environment, separate from Windows Python | Windows DLL wheels cannot be used in WSL. No public matched set yet. |
| Linux / CUDA server | Separate hardware and precision acceptance | SM89 PC evidence does not qualify A100 / SM80. |

## Install the Mac preview

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install "diffusor-rt[compiler,mlx-local,serve,gui] @ https://github.com/jimmy-yilong-li/DiffusorRT/releases/download/v0.2.6/diffusor_rt-0.2.6-py3-none-any.whl"
diffusorrt run "dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1" --prompt "Hello!"
diffusorrt gui "dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1"
```

The window is included in the SDK. A separate native Swift App archive is
not included. Do not look for an App zip or CUDA wheel in these release assets.

## Do not mix release sets

- `py3-none-any` describes the SDK payload, not GPU compatibility.
- Native-runtime and SDK version numbers need not be equal. A CUDA release
  must identify the exact tested pair, Python ABI, OS/CPU and GPU targets.
- Do not attach an older private `diffusorrt_runtime` wheel to this SDK merely
  because it imports. The private device-session arguments have changed.
- Windows, Linux/WSL and macOS native binaries are not interchangeable. Keep
  their virtual environments separate; changing `LD_LIBRARY_PATH` does not
  convert a Windows DLL into a Linux library.
- Keep model caches and conversations when upgrading. Use the new wheel URL
  with `--upgrade`; do not delete user history to resolve a package mismatch.

Earlier previews are superseded. The previous 0.2.5 release is the rollback
reference for the SDK-only path; CUDA rollback requires its own complete pair.
Historical release notes are not installation recommendations.

## Verify the download

File: `diffusor_rt-0.2.6-py3-none-any.whl`

```text
86c9cee779bf65e127e3d5326caed806cf7bdada0d080ae41a5d540347116acb
```

Compare with `shasum -a 256 <file>` on Mac, `sha256sum <file>` on Linux, or
`Get-FileHash <file> -Algorithm SHA256` in PowerShell.
Installing the base SDK alone does not install model preparation or GPU extras.

## Qualification scope

Mac, CUDA PC and CUDA server releases have independent evidence. This release
does not promote CUDA, renew historical model receipts or claim new model
accuracy/performance. The SDK is Apache-2.0; private binaries and model weights
retain their own terms.
