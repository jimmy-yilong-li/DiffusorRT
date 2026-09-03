# DiffusorRT Copyright, Licensing, and Distribution Policy

**Date:** 2026-07-30  
**Last revised:** 2026-08-14
**Status:** binding product and repository constraint
**Scope:** source repositories, packages, native binaries, model artifacts,
documentation, and release claims

> This document records the project's engineering and distribution decision. It
> is not the final license text, an EULA, or legal advice. Qualified counsel
> must review the final public-export boundary, copyright notices, third-party
> obligations, and binary license before public distribution.

## 1. Binding Decision

DiffusorRT is **not** planned as a fully open-source implementation.

The product will use a clean two-part distribution model:

1. an **open-source SDK and executable reference implementation**, licensed
   under Apache License 2.0; and
2. a **proprietary optimized native runtime**, distributed only as compiled
   platform binaries under a separate DiffusorRT Runtime Binary License or
   commercial agreement.

Model weights, tokenizers, and backend-compiled model artifacts remain subject
to their own upstream licenses. DiffusorRT does not acquire or transfer model
rights merely by packaging or loading an artifact.

Public descriptions must use precise language:

> **DiffusorRT provides an open-source SDK and reference runtime with
> proprietary optimized native runtime binaries.**

The project must not describe the entire implementation or optimized runtime as
open source.

## 2. Current Repository State

The complete maintained source repository is the private
[`DiffusorRT-Source`](https://github.com/jimmy-yilong-li/DiffusorRT-Source).
Its `main` branch is the sole development and release source of truth.

The root `LICENSE`, `python/LICENSE`, and Python package metadata identify
Apache-2.0. Those notices remain effective for the files and copies to which
they apply. They must not be treated as a blanket license for future files
explicitly classified as proprietary; those files require their own notices.

The proprietary source surface and public export are not yet complete:

- it does not revoke rights already granted for any copy previously distributed
  under Apache-2.0;
- proprietary native source must not be added to an Apache-2.0 public
  repository;
- proprietary source must be classified and excluded by a fail-closed public
  exporter before the public repository or any source release is created;
- release metadata must describe each artifact's actual license instead of
  treating this monorepo's current metadata as the final product layout.

Until that export boundary and legal review are complete, DiffusorRT is **not
release-ready under the two-part model**.

## 3. Public SDK Boundary

The public `diffusor-rt` project should remain useful for integration,
correctness, and conformance without access to proprietary source. Its intended
Apache-2.0 surface includes:

- Python APIs, CLI surfaces, and native binding declarations;
- the stable public C ABI headers and version-negotiation contract;
- execution-package schemas, GraphIR contracts, model-adapter interfaces, and
  backend capability manifests;
- the portable NumPy/reference runtime and reference policy implementations;
- the **Reference Compiler** (also: Reference Package Builder) — checkpoint
  resolution and pinning, licence propagation, disk budgeting, download,
  conversion to an execution package, and cache publication with receipts;
- package validation, diagnostics, conformance tests, and public golden vectors;
- examples, user documentation, integration guides, and plugin contracts.

The public reference path is the semantic oracle. It need not provide the
throughput of the proprietary runtime, but it must be real, runnable, and
honest about its capabilities.

`Engine.from_pretrained()` therefore resolves entirely within the Apache-2.0
surface. A public SDK whose primary entry point required a proprietary binary
would not be an independently runnable reference path, which is the whole point
of the split.

## 4. Proprietary Native Boundary

The boundary is a **per-file surface**, not a separate source repository.
Proprietary source lives on the `private` surface of `DiffusorRT-Source` and
reaches users only as a separately licensed compiled binary.

The `private` surface may contain:

- the optimized C++ runtime and session implementation;
- compiled decode scheduling, cache, paging, and memory-planning machinery;
- the **Optimizing Native Compiler**: graph optimization, backend lowering,
  quantization and tuning, device-specific execution plans, and signed backend
  artifacts. This is the compiler that is proprietary. Resolving a checkpoint
  into an execution package is *not* — that is the public Reference Compiler
  above, and calling it "the compiler" in a proprietary list protects no
  optimization technique while making the public SDK depend on a private binary;
- Metal, CUDA, TensorRT, and future mobile backend implementations;
- custom kernels, tuning databases, device-specific heuristics, and production
  profiling logic;
- signing, update, entitlement, and commercial integration components.

These sources carry a proprietary copyright notice and live on the `private`
surface of `DiffusorRT-Source`. They must not be copied into, generated into,
or vendored through the public Apache-2.0 repository. That boundary is a
per-file classification in `repo_manifest.toml` rather than a second
repository. The classification is a PRECONDITION, not the enforcement: no
deterministic exporter exists yet, so nothing has been produced from it or
verified against it.

The proprietary implementation may depend on the public ABI. The public SDK
must not import private Python source or depend on private implementation
details.

## 5. Binary Distribution

The optimized runtime should ship as platform-specific binary artifacts, for
example:

```text
diffusor-rt                 # Apache-2.0 SDK and reference runtime
diffusorrt-apple            # proprietary macOS native binary wheel
diffusorrt-cuda             # proprietary Linux/CUDA native binary wheel
libdiffusorrt_core.dylib    # proprietary native runtime
libdiffusorrt_core.so       # proprietary native runtime
*.drtbundle                 # model/backend artifact; license recorded per bundle
```

The binary license must be accepted or otherwise made legally available before
the proprietary runtime is loaded. The final license must explicitly settle:

- free, evaluation, research, and commercial-use rights;
- redistribution and OEM/embedded rights;
- device, user, or organization limits, if any;
- reverse-engineering and modification restrictions to the extent permitted by
  applicable law;
- update, support, warranty, liability, and termination terms;
- offline use, telemetry, privacy, and export-control behavior;
- whether compiled model bundles may be redistributed.

The recommended product structure is a broadly usable local runtime binary plus
separate commercial/OEM terms for redistribution, embedding, support, and
custom platform delivery. This commercial policy is not final until approved
by the copyright owner and counsel.

## 6. ABI And Packaging Constraint

The public/private boundary is an ABI, not an internal module import.

The public contract must define:

- ABI version and compatibility negotiation;
- opaque engine, model, session, request, tensor, and cache handles;
- explicit ownership, lifetime, thread-safety, cancellation, and error rules;
- tensor dtype, shape, stride, memory-location, and synchronization semantics;
- capability discovery and fail-closed backend selection;
- artifact identity, signature, hash, and version compatibility;
- deterministic conformance vectors for reference-versus-native parity.

The SDK discovers and loads a signed, compatible platform binary. Missing,
incompatible, unsigned, or unlicensed binaries must produce an actionable
error or select the public reference backend. They must never trigger a silent,
behavior-changing fallback.

## 7. Repository And Build Separation

The approved layout is one private source repository and one public
distribution repository, not two source repositories. `repo.md` defines the
topology; `repo_manifest.toml` records the per-file storage boundary and is
checked on every push. Producing the public repository from it is `repo.md`
section 8, and is not built.

| Repository | Visibility | License | Responsibility |
|---|---|---|---|
| `DiffusorRT-Source` | private | mixed, per-file surface | Complete source: the `public` surface below, the `private` surface, plus research and evidence |
| `DiffusorRT` | public | Apache-2.0 source + separately licensed binaries | Exported `public` surface: APIs, ABI headers, schemas, reference runtime, **Reference Compiler**, conformance, docs — plus platform binaries built from the private surface, each carrying its own licence |

What must never happen is unchanged: proprietary source never enters the
public repository. Today every file carries a classification saying which side
it is on, and the push check refuses an unclassified one. That is a ledger, not
an enforced export -- calling it "blocked" would claim an outcome nothing has
yet produced. Measured on the current classification, a naive export would also
leave public files depending on excluded ones, which the exporter has to
resolve.

"Compiler" unqualified is ambiguous and must not be used in licensing prose:
the Reference Compiler is public, the Optimizing Native Compiler is private.

Build and release systems may combine artifacts, but source ownership and
license metadata must remain separate. Generated wheels must contain only files
permitted by their declared license.

## 8. Copyright And Contributions

Before public release, the project must select the exact legal copyright holder
name: either the owner's legal name or an owning company. Do not publish a
placeholder copyright notice.

The expected notices are:

```text
Public SDK:
  Copyright (c) <year> <legal owner>
  Licensed under the Apache License, Version 2.0.

Private source surface:
  Copyright (c) <year> <legal owner>
  All rights reserved.
```

The public SDK needs an inbound-contribution policy before accepting external
changes. The project must deliberately choose DCO, CLA, or another counsel-
approved mechanism. Contributions to the private surface require written
employment, contractor, or IP-assignment terms. A public contribution must be
reviewed and integrated into `DiffusorRT-Source/main`, then returned through
the exporter; it must not create a second public implementation.

All imported, generated, or AI-assisted code remains subject to source and
license review. The project must not incorporate code copied from incompatible
projects merely because an implementation was used as an engineering
reference.

## 9. Third-Party And Model Licenses

Every release must include a machine-readable dependency inventory and the
required third-party notices. In particular:

- Apache-2.0, MIT, BSD, and other permissive dependencies retain their notices;
- copyleft, source-available, research-only, non-commercial, and custom
  licenses require explicit compatibility review before inclusion;
- Apple, MLX, CUDA, TensorRT, and other vendor components may be used or
  redistributed only under their respective terms;
- model, tokenizer, adapter, and dataset licenses are recorded independently
  from the DiffusorRT software license;
- when model redistribution is not permitted, DiffusorRT must support local
  acquisition and compilation instead of bundling the model.

Each `.drtbundle` or equivalent deployment artifact must record source model,
revision, tokenizer identity, transformation provenance, and applicable license
metadata.

## 10. Release Claims

Allowed:

- "DiffusorRT is an edge-first dLLM inference framework."
- "The DiffusorRT SDK and reference runtime are Apache-2.0."
- "Optimized native runtimes are distributed as proprietary binaries."
- "The SDK can run the reference backend without the proprietary runtime."

Not allowed:

- "DiffusorRT is fully open source."
- "All DiffusorRT backends and kernels are Apache-2.0."
- "The Apache license covers model weights or vendor runtimes."
- "A proprietary binary is open source because its ABI is public."

## 11. Release Blockers

No public product release under the two-part model may be declared complete
until all of the following are done:

1. create the public `DiffusorRT` repository and implement a deterministic,
   clean-history export from `DiffusorRT-Source`;
2. select and record the legal copyright holder;
3. obtain legal review of the source/export boundary and binary EULA;
4. retain Apache-2.0 only on the public SDK files and artifacts;
5. add proprietary notices and enforce access controls for the private source
   surface;
6. align package metadata, wheel contents, documentation, and download terms;
7. publish ABI compatibility and binary-support matrices;
8. generate third-party notices and a dependency/software bill of materials;
9. implement model-license checks and artifact provenance;
10. decide public contribution, commercial, OEM, support, and redistribution
    policies;
11. sign release binaries and publish hashes;
12. audit all public claims against Section 10.

## 12. Change Control

This file is a binding architecture and release constraint. Any proposal that
changes which source is public, which binaries are proprietary, or what rights
are granted must:

1. update this document;
2. update `repo.md`, `plan.md`, `progress.md`, and
   `docs/product/README.md`; update `design.md` only when runtime architecture,
   rather than repository placement, changes;
3. identify migration effects on existing license grants and released
   artifacts;
4. receive explicit owner approval and legal review before distribution.

Historical documents remain point-in-time records. New implementation plans
must link to this policy and may not silently cross the public/private boundary.
