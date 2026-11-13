# DiffusorRT SDK source

This directory contains the Apache-2.0 Python SDK for **0.2.6**: API, CLI,
model preparation, reference execution, public MLX execution, conversations,
and the Mac and Windows chat interfaces.

The 191 files under `src/` match the runtime payload of the published 0.2.6
wheel byte-for-byte. [EXPORT_RECEIPT.json](../EXPORT_RECEIPT.json) records
the selected source files and hashes. The wheel itself has not changed.

## Install

Use a Python 3.11 environment. From the repository root on Apple Silicon:

```bash
python -m pip install -e "./python[compiler,mlx-local,gui]"
python -c "import diffusor_rt; print(diffusor_rt.__version__)"
python -m diffusor_rt --help
```

For the base SDK on Windows or Linux:

```bash
python -m pip install -e ./python
```

Base installation supplies the API, CLI and package tools. It does not add
CUDA libraries or a native GPU provider. See the [Mac guide](../mac/README.md)
for the public model path and [Windows guide](../windows/README.md) for
current CUDA availability. Do not install `mlx-local` on Windows.

To build a wheel without downloading runtime dependencies:

```bash
python -m pip wheel --no-deps ./python
```

The build frontend obtains the pinned build backend separately. Installing
from source does not download model weights; the first `run` or `pull`
prepares the selected model.

## Find the implementation

| Component | Source |
|---|---|
| Public API and command entry points | [diffusor_rt](./src/diffusor_rt) |
| Engine and conversations | [engine.py](./src/edllm/engine.py), [conversation.py](./src/edllm/conversation.py) |
| Model preparation | [compiler](./src/edllm/compiler) |
| Reference and public MLX execution | [backends](./src/edllm/backends) |
| Mac page, controller, WebKit window and host protocol | [gui/mac](./src/edllm/gui/mac) |
| Windows controller and local page server | [gui](./src/edllm/gui) |
| Windows HTML, CSS and JavaScript | [gui/web](./src/edllm/gui/web) |

The two desktop interfaces remain separate implementations over the same SDK.
The Mac window is a WebKit-hosted page. The Windows interface uses a local
page in Edge; it is not an MSI or WinUI application.

## Download without Git

Download [diffusor-rt-sdk-0.2.6-source.tar.gz](https://github.com/jimmy-yilong-li/DiffusorRT/releases/download/v0.2.6/diffusor-rt-sdk-0.2.6-source.tar.gz)
and compare its hash with [SHA256SUMS-sdk-0.2.6.txt](https://github.com/jimmy-yilong-li/DiffusorRT/releases/download/v0.2.6/SHA256SUMS-sdk-0.2.6.txt).
After extraction, run the install commands from its top-level directory.
Use this named SDK archive: GitHub's automatic archive of the original
`v0.2.6` tag predates the SDK source publication.

## Not included

This source package does not include proprietary native C++/CUDA execution,
native runtime wheels, the separate Swift Mac App, model weights, development
logs or private experiment data. It is a runnable SDK source distribution,
not the complete development repository.

Source publication adds no new hardware or model qualification. The
[0.2.6 installation guide](../docs/product/sdk-0.2.6-installation.md) describes
the supported artifact combinations. The [Apache-2.0 license](./LICENSE)
covers this SDK; models and third-party dependencies retain their own licenses.
