# Licensing and distribution

Updated: 2026-11-13.

This page describes the files and downloads in the public DiffusorRT
repository. It is not a replacement for the licenses themselves and does not
grant rights to unpublished components.

## Public SDK

The published Python SDK, Reference Compiler, NumPy reference implementation,
public MLX path, and Python/HTML/CSS/JavaScript chat interfaces are licensed
under [Apache License 2.0](./LICENSE). The Python package carries the same
[license](./python/LICENSE).

SDK 0.2.6 is available as:

- the versioned Python wheel in [Releases](https://github.com/jimmy-yilong-li/DiffusorRT/releases/tag/v0.2.6);
- the source under [python/](./python/README.md), including the Mac and Windows
  page interfaces;
- an explicitly named SDK source archive attached to that release.

The exported runtime files match the wheel. The source export receipt records
the selected files, source revision and hashes; it is not a certificate of
model accuracy or platform support.

## Separate components

Optimized native runtime implementations are not part of this public source
release. Publishing an SDK interface to a native provider does not publish
that provider or grant an Apache-2.0 license to it.

SDK 0.2.6 includes no proprietary CUDA/native wheel, model weights or standalone
Swift Mac App. Future native binaries require their own approved distribution
terms and platform acceptance. The [installation guide](./docs/product/sdk-0.2.6-installation.md)
lists the combinations currently available.

Do not describe the whole DiffusorRT implementation as open source. The
published SDK is open source; unpublished or separately licensed components
remain outside that grant. Existing rights granted to previously distributed
Apache-2.0 files are not revoked by this boundary.

## Third-party dependencies and models

Dependencies retain their own licenses and notices. Installing an SDK extra
obtains third-party packages separately; the SDK license does not relicense
MLX, PyTorch, CUDA libraries or other dependencies.

Model weights, tokenizers and compiled model artifacts remain subject to
their upstream terms. Downloading or converting a model through DiffusorRT
does not change those rights. Check the model's license before using or
redistributing its files.

This source archive contains no model checkpoints or bundled third-party
native libraries. The Python package metadata lists its dependencies; any
future redistribution of dependencies must preserve their required notices
and satisfy their terms.

## Contributions and notices

See [CONTRIBUTING.md](./CONTRIBUTING.md) before submitting a contribution.
Preserve existing copyright and license notices. Do not submit third-party
code without the necessary rights, or treat generated code as exempt from
license review.

This source publication changes availability, not the Apache-2.0 license
text or the licensing status of private components.
