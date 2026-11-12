<div align="center">

<img src="../docs/product/assets/diffusorrt-logo.png" alt="DiffusorRT" width="110">

<h1>DiffusorRT for Windows</h1>
<h3>Bringing local diffusion to NVIDIA PCs.</h3>

<p>
<img src="https://img.shields.io/badge/CUDA_PC-In_development-7c3aed?style=flat-square" alt="CUDA PC in development">
<img src="https://img.shields.io/badge/Prototype-RTX_4070_%C2%B7_SM89-0891b2?style=flat-square" alt="RTX 4070 / SM89 prototype">
</p>

<p><a href="#desktop-experience">Desktop experience</a> ·
<a href="#availability">Availability</a> · <a href="#windows-and-wsl">Windows &amp; WSL</a> ·
<a href="../mac/README.md">Mac</a> · <a href="../README.md">All platforms</a></p>

</div>

The CUDA PC prototype brings DiffusorRT's shared model APIs and conversations
to a local Windows chat interface.

## Desktop experience

The interface is a local page in a standalone **Microsoft Edge window**,
not a WinUI application or Windows installer.

- **Masked-text preview:** see positions being filled as a reply develops.
- **Local conversations:** return to saved history shared with the CLI.
- **Cooperative Stop:** supported cache-free execution stops at decode boundaries.
- **Resource panel:** inspect model, precision and named GPU memory readings.

Real Windows/RTX 4070 runs have been reported during development. A
Windows-captured screenshot is not available in this public update; the Mac
screenshots on the project homepage show a different application.

## Availability

**No complete public Windows CUDA artifact set is available yet.**
The downloadable [SDK 0.2.5](https://github.com/jimmy-yilong-li/DiffusorRT/releases/tag/v0.2.5)
does not include a Windows CUDA provider. Installing it alone does not enable
NVIDIA inference.

The first Windows preview will publish its compatible SDK/native wheels,
CUDA dependencies, supported model/precision combinations and installation
instructions together. The Mac 0.2.6 preview is not a Windows CUDA release.

## Windows and WSL

| Environment | Required execution artifact |
|---|---|
| Native Windows | Windows native wheel and DLL dependencies for the supported Python/CPU ABI. |
| WSL2 | Linux native wheel and Linux libraries; Windows DLLs do not apply. |
| Linux CUDA server | Separate server delivery with its advertised GPU targets and model support. |

RTX 4070/SM89 and A100/SM80 are different hardware targets. Results on one
machine do not establish support on another. The public SDK's `py3-none-any`
tag describes Python packaging, not universal GPU compatibility.

Conversations use plain, unencrypted SQLite. The interface uses loopback access;
the project is focused on local inference, not Internet-facing multi-user serving.

## News

**2026-11-12 — Windows CUDA PC preview in development.**
The prototype includes a masked-text view, shared conversations and resource
inspection. No Windows installer or CUDA wheel is released with this page.

Follow [project News](../README.md#news) and
[Releases](https://github.com/jimmy-yilong-li/DiffusorRT/releases) for downloads.

## License

The public SDK is [Apache-2.0](../LICENSE). Native runtime distribution and model
assets have separate terms; this page grants no license to unpublished binaries.
