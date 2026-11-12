# DiffusorRT — user guide

Start with the [project homepage](./README.md) and the
[SDK 0.2.6 download](https://github.com/jimmy-yilong-li/DiffusorRT/releases/tag/v0.2.6).

| I want to… | Open |
|---|---|
| Install the Python library | [Install](./README.md#install) |
| Chat on Apple Silicon and see diffusion | [Mac guide and real screenshots](./mac/README.md) |
| Understand Windows, WSL and CUDA availability | [Windows guide](./windows/README.md) |
| Avoid incompatible SDK/native combinations | [Matching installations](./docs/product/sdk-0.2.6-installation.md) |
| Call the API from a script | [Python API](./README.md#python-api) · [Quickstart](./examples/quickstart.py) |
| Use CLI, history, model management or HTTP | [Command line](./README.md#command-line) · [Local service](./README.md#model-manager-and-local-service) |
| Check model support and limitations | [Models and platforms](./README.md#supported-models-and-platforms) |
| Read changes, license or citation | [News](./README.md#news) · [License](./LICENSE) · [Citation](./CITATION.cff) |
| Report a problem | [Issues](https://github.com/jimmy-yilong-li/DiffusorRT/issues) · [Security](./SECURITY.md) |

This is a distribution repository, not a complete development checkout.
Install the wheel to run DiffusorRT. The selected API excerpts are reading
examples; they are not a standalone editable source installation.

The public release does not include native CUDA wheels or the separate Swift
App. Mac results do not qualify CUDA PC or CUDA server.
