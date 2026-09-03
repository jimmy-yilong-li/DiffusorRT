# Contributing

DiffusorRT is an edge-first diffusion language-model runtime. Contributions should
preserve its fail-closed correctness, explicit backend capabilities, and
evidence-backed performance claims.

## Development Setup

Create an isolated Python 3.11 or newer environment, then install the portable
development dependencies:

```bash
python -m pip install -e "./python[test]"
```

Install optional extras only for the backend or converter you are changing:

```bash
python -m pip install -e "./python[mlx-local]"
python -m pip install -e "./python[converter]"
```

## Change Requirements

1. Keep the portable control plane importable without MLX, Torch, or
   Transformers.
2. Add a regression test before fixing a behavior defect.
3. Preserve the full-forward oracle and fail closed when exact reuse cannot be
   proven.
4. Do not claim support for a platform, model family, cache mode, or kernel
   without its declared conformance and product gate.
5. Describe any queue, contract, or capability change in the handoff. The
   integrator updates `plan.md`, `progress.md`, and user documentation once,
   after evidence and merge order are settled.
6. Never commit model weights, secrets, private package paths, or generated
   benchmark artifacts that are not intentionally part of the evidence record.

Any change to `_d2f_exact_forward_from_weights` or
`_transformer_layer_forward_d2f_exact` must state the gated behavior in the
commit message and include focused parity evidence. Mathematical changes also
require an explicit contract update in `design.md` or `plan.md` and an oracle
comparison. Do not create a separate active plan or spec.

## Verification

Run the smallest relevant test first, then the portable suite:

```bash
PYTHONPATH=python/src pytest -q path/to/focused_test.py
PYTHONPATH=python/src pytest -q
git diff --check
```

Backend-specific changes must also run their opt-in maintainer gate on declared
hardware. Report exact commands and outcomes; do not translate skipped hardware
tests into support claims.

## Pull Requests

Keep commits reviewable and avoid mixing runtime behavior, research probes, and
documentation-only direction changes. A pull request should state:

- the user-visible or contract change;
- the supported and unsupported paths;
- the tests and hardware used;
- performance evidence, if performance is claimed;
- rollback or fallback behavior for optional accelerated paths.

By submitting a contribution, you agree that it is licensed under Apache-2.0.
