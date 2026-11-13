from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from edllm.graph import GraphIR


def validate_package_version(package: dict) -> list[str]:
    """Validate ONLY the package version block (major/minor/patch presence).

    Full manifest/schema validation runs at load time (``model.load_runtime`` ->
    ``_validate_manifest_schema``); this is a shallow version pre-check — hence the
    narrow name. It does NOT validate the rest of the package.
    """
    errors: list[str] = []
    version = package.get("version")
    if not isinstance(version, dict):
        return ["version missing"]
    for key in ("major", "minor", "patch"):
        if key not in version:
            errors.append(f"{key} missing from version")
    return errors


def is_compatible(runtime_major: int, package_major: int) -> bool:
    return runtime_major == package_major


def write_execution_package(
    output_dir: str | Path,
    manifest: dict,
    tokenizer_manifest: dict | None = None,
    special_tokens_map: dict | None = None,
    weights: bytes = b"placeholder-weights",
    weights_filename: str = "weights.bin",
    graph: GraphIR | None = None,
    graph_filename: str = "graph.json",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8", newline="\n")
    if tokenizer_manifest is not None:
        tokenizer = Tokenizer(
            WordLevel(
                vocab={token: int(token_id) for token, token_id in tokenizer_manifest["vocab"].items()},
                unk_token="<unk>",
            )
        )
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.save(str(output_dir / "tokenizer.json"))
    if special_tokens_map is not None:
        (output_dir / "special_tokens_map.json").write_text(json.dumps(special_tokens_map, indent=2), encoding="utf-8", newline="\n")
    if weights is not None:
        (output_dir / weights_filename).write_bytes(weights)
    if graph is not None:
        (output_dir / graph_filename).write_text(graph.to_json(indent=2), encoding="utf-8", newline="\n")
    return output_dir


def read_manifest(package_dir: str | Path) -> dict:
    package_dir = Path(package_dir)
    return json.loads((package_dir / "model.json").read_text(encoding="utf-8"))
