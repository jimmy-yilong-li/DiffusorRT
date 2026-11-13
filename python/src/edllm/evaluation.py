"""Shared generative lm-eval tasks and a measured public-Engine adapter.

No model or dataset is acquired on import. The optional harness is required
only when evaluating. Source revisions and selected row IDs are the persistent
input; question documents stay in the external dataset cache.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import time
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any

from .benchmark import (
    SCORING_PROTOCOL, EvaluationCase, _evaluation_row, _measure_engine_request,
    _process_peak_rss_bytes, _record_not_attempted, score_evaluation_answer, write_dataset_evaluation_report,
)
from .engine import Engine
from .model import _close_preserving_primary

HARNESS_VERSION = "0.4.9.1"
HARNESS_COMMIT = "03c44adc0586f88bb343a74da1a1c602103536dd"
HARNESS_SOURCE = "https://github.com/ZHZisZZ/lm-evaluation-harness"
TASK_NAMES = ("arc_easy_gen", "commonsense_qa_gen", "winogrande_gen", "triviaqa", "svamp_gen")
SVAMP_ERRATA = {
    "5be77703a6d891ae476d7c082787ad361392aa02453b132516cdd5f4e7934e3e": {
        "chal-680": "Equation (4 - 2) + 3 evaluates to 5, but Answer is 1.",
    },
}


def _require_harness() -> None:
    try:
        installed = distribution("lm_eval")
    except PackageNotFoundError as error:
        raise ImportError(f"Evaluation requires lm_eval from {HARNESS_SOURCE}@{HARNESS_COMMIT}") from error
    origin = json.loads(installed.read_text("direct_url.json") or "{}")
    if (installed.version != HARNESS_VERSION
            or origin.get("vcs_info", {}).get("commit_id") != HARNESS_COMMIT):
        raise ValueError(f"Install the shared harness from {HARNESS_SOURCE}@{HARNESS_COMMIT}")


@lru_cache(maxsize=1)
def _triviaqa_config() -> dict[str, Any]:
    _require_harness()
    import lm_eval
    import yaml

    path = Path(lm_eval.__file__).parent / "tasks" / "triviaqa" / "default.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _stop_at(text: str, until: list[str] | str) -> str:
    stops = [until] if isinstance(until, str) else until
    positions = [text.find(stop) for stop in stops if stop and stop in text]
    return text[:min(positions)] if positions else text


def score_triviaqa_answer(text: str, aliases: tuple[str, ...]) -> tuple[str | None, bool]:
    """Use the installed task's stops, filters and native exact-match metric."""
    config = _triviaqa_config()
    from lm_eval.api.instance import Instance
    from lm_eval.api.metrics import exact_match_fn
    from lm_eval.filters import build_filter_ensemble

    instance = Instance("generate_until", {}, ("", {}), 0)
    instance.resps = [_stop_at(text, config["generation_kwargs"]["until"])]
    for pipeline in config["filter_list"]:
        components = [(f["function"], {k: v for k, v in f.items() if k != "function"})
                      for f in pipeline["filter"]]
        build_filter_ensemble(pipeline["name"], components).apply([instance])
    parsed = instance.filtered_resps[config["filter_list"][0]["name"]]
    if not parsed or not aliases:
        return None, False
    metric = config["metric_list"][0]
    score = exact_match_fn(
        references=list(aliases), predictions=[parsed] * len(aliases),
        ignore_case=metric["ignore_case"], ignore_punctuation=metric["ignore_punctuation"],
    )["exact_match"]
    return parsed, bool(score > 0)


def choice_prompt(doc: dict[str, Any]) -> str:
    choices = doc["choices"]
    options = "\n".join(f"{label}. {text}" for label, text in zip(
        choices["label"], choices["text"], strict=True,
    ))
    return f"Question: {doc['question']}\n{options}\nReply with one choice label.\nAnswer:"


def winogrande_prompt(doc: dict[str, Any]) -> str:
    return (
        f"Complete the blank in the sentence.\n{doc['sentence']}\n"
        f"1. {doc['option1']}\n2. {doc['option2']}\n"
        "Reply with one choice label.\nAnswer:"
    )


def svamp_prompt(doc: dict[str, Any]) -> str:
    return (
        f"{doc['Body']}\n{doc['Question']}\n"
        "End with Answer: followed by your numeric answer, without brackets."
    )


def _case(task: str, doc: dict[str, Any], doc_id: int, prompt: str) -> EvaluationCase:
    if task in ("arc_easy_gen", "commonsense_qa_gen"):
        return EvaluationCase(task, doc_id, doc["id"], prompt, "choice",
                              (doc["answerKey"],), 32, tuple(doc["choices"]["label"]),
                              tuple(doc["choices"]["text"]))
    if task == "winogrande_gen":
        return EvaluationCase(task, doc_id, None, prompt, "choice", (doc["answer"],),
                              32, ("1", "2"), (doc["option1"], doc["option2"]))
    if task == "svamp_gen":
        return EvaluationCase(task, doc_id, doc["ID"], prompt, "numeric",
                              (str(doc["Answer"]),), 128)
    if task == "triviaqa":
        return EvaluationCase(task, doc_id, doc["question_id"], prompt, "exact",
                              tuple(doc["answer"]["aliases"]), 32)
    raise ValueError(f"Unknown evaluation task: {task}")


def _choice_results(doc: dict[str, Any], results: list[str]) -> dict[str, int]:
    case = _case("arc_easy_gen", doc, 0, "")
    return {"acc": int(score_evaluation_answer(case, results[0])[1])}


def _winogrande_results(doc: dict[str, Any], results: list[str]) -> dict[str, int]:
    return {"acc": int(score_evaluation_answer(_case("winogrande_gen", doc, 0, ""), results[0])[1])}


def _svamp_results(doc: dict[str, Any], results: list[str]) -> dict[str, int]:
    return {"acc": int(score_evaluation_answer(_case("svamp_gen", doc, 0, ""), results[0])[1])}


def task_definitions(selection: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """The same task definitions are passed to every backend, without checkpoints."""
    definitions = {
        "arc_easy_gen": ("allenai/ai2_arc", "ARC-Easy", "test", choice_prompt, _choice_results),
        "commonsense_qa_gen": ("tau/commonsense_qa", None, "validation", choice_prompt, _choice_results),
        "winogrande_gen": ("allenai/winogrande", "winogrande_xl", "validation", winogrande_prompt, _winogrande_results),
        "svamp_gen": ("json", None, "test", svamp_prompt, _svamp_results),
    }
    configs = []
    for name, source in selection.items():
        if name not in TASK_NAMES:
            raise ValueError(f"Unknown evaluation task: {name}")
        ids = source["doc_ids"]
        # This pinned harness iterates selected rows in source order. An empty
        # selection would otherwise fall through to the whole split.
        if (not ids or any(type(i) is not int or i < 0 for i in ids)
                or ids != sorted(set(ids))):
            raise ValueError(f"{name}: doc_ids must be nonempty, increasing, unique row indices")
        if name != "svamp_gen" and not re.fullmatch(r"[0-9a-f]{40}", source["revision"]):
            raise ValueError(f"{name}: revision must pin a dataset commit, not a mutable branch")
        if name == "triviaqa":
            config = {**_triviaqa_config(), "dataset_kwargs": {"revision": source["revision"]}}
            config["generation_kwargs"] = {**config["generation_kwargs"], "max_gen_toks": 32}
        else:
            path, subset, split, prompt, scorer = definitions[name]
            dataset_kwargs = {"revision": source["revision"]} if name != "svamp_gen" else {
                "data_files": {"test": source["path"]},
            }
            if name == "svamp_gen":
                digest = hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest()
                if digest != source["sha256"]:
                    raise ValueError("svamp_gen: source-file digest differs from frozen selection")
            config = {
                "task": name, "dataset_path": path, "dataset_name": subset,
                f"{split}_split": split, "dataset_kwargs": dataset_kwargs,
                "output_type": "generate_until", "doc_to_text": prompt,
                "doc_to_target": "answer" if name == "winogrande_gen" else (
                    "Answer" if name == "svamp_gen" else "answerKey"
                ),
                "process_results": scorer, "num_fewshot": 0, "repeats": 1,
                "generation_kwargs": {"until": [], "do_sample": False, "temperature": 0.0,
                                      "max_gen_toks": 128 if name == "svamp_gen" else 32},
                "metric_list": [{"metric": "acc", "aggregation": "mean", "higher_is_better": True}],
                "metadata": {"version": 5.0},
            }
        configs.append(config)
    if not configs:
        raise ValueError("Select at least one task; no implicit full-dataset evaluation")
    return configs


def _sample_question_rows(
    documents: Any, *, count: int, rng: random.Random, id_field: str,
    excluded_ids: dict[str, str],
) -> list[int]:
    candidates = {}
    for index, doc in enumerate(documents):
        identity = doc.get(id_field)
        if identity is None:
            identity = f"row-{index:012d}"
        if identity not in excluded_ids:
            candidates.setdefault(identity, index)
    if count > len(candidates):
        raise ValueError(f"only {len(candidates)} unique eligible questions")
    identities = rng.sample(sorted(candidates), count)
    return sorted(candidates[identity] for identity in identities)


def freeze_harness_selection(
    sources: dict[str, dict[str, Any]], *, questions_per_task: int, seed: int,
) -> dict[str, dict[str, Any]]:
    """Freeze unique question IDs from pinned task sources, without inference.

    Duplicate evidence rows share one question identity. Known source-bound
    label errata are excluded explicitly, never silently corrected in place.
    The returned row mapping can be saved and passed to any harness backend.
    """
    if type(questions_per_task) is not int or questions_per_task <= 0:
        raise ValueError("questions_per_task must be a positive integer")
    _require_harness()
    from lm_eval.api.task import ConfigurableTask

    specs = task_definitions({name: {**source, "doc_ids": [0]}
                              for name, source in sources.items()})
    tasks = {spec["task"]: ConfigurableTask(config=spec) for spec in specs}
    rng = random.Random(seed)
    selected = {}
    id_fields = {"triviaqa": "question_id", "svamp_gen": "ID",
                 "arc_easy_gen": "id", "commonsense_qa_gen": "id"}
    for name in TASK_NAMES:
        if name not in sources:
            continue
        source = {key: value for key, value in sources[name].items() if key != "doc_ids"}
        errata = SVAMP_ERRATA.get(source.get("sha256"), {}) if name == "svamp_gen" else {}
        indices = _sample_question_rows(
            tasks[name].eval_docs, count=questions_per_task, rng=rng,
            id_field=id_fields.get(name, ""), excluded_ids=errata,
        )
        selected[name] = {
            **source, "doc_ids": indices,
            "selection_policy": "unique-question-id-v1", "excluded_ids": sorted(errata),
        }
    return selected


def _json_value(value: Any) -> Any:
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Not a JSON value: {type(value).__name__}")


def _load_harness_cases(selection: dict[str, dict[str, Any]]) -> tuple[list, dict, list[EvaluationCase]]:
    from lm_eval.api.task import ConfigurableTask

    specs = task_definitions(selection)
    tasks = {spec["task"]: ConfigurableTask(config=spec) for spec in specs}
    cases = [
        _case(name, tasks[name].eval_docs[index], index, "")
        for name, source in selection.items() for index in source["doc_ids"]
    ]
    seen = set()
    for case in cases:
        identity = (case.task, case.id if case.id is not None else case.doc_id)
        if identity in seen:
            raise ValueError(f"{case.task}: selection repeats question identity {case.id!r}")
        seen.add(identity)
        if case.task == "svamp_gen":
            errata = SVAMP_ERRATA.get(selection[case.task]["sha256"], {})
            if case.id in errata:
                raise ValueError(f"svamp_gen: known label erratum {case.id}: {errata[case.id]}")
    return specs, tasks, cases


def rescore_harness_evaluation(*, source_dir: str | Path, output_dir: str | Path) -> Path:
    """Score saved responses under the current protocol without model execution.

    Original responses, timings and files stay unchanged. Pinned datasets supply
    option text and aliases; the output records both the original run's hashes
    and the current scoring code. This is a derivation, not a new benchmark.
    """
    _require_harness()
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    inputs = {name: (source_dir / name).read_bytes()
              for name in ("run.json", "selection.json", "rows.jsonl")}
    original = json.loads(inputs["run.json"])
    selection = json.loads(inputs["selection.json"])
    _, _, cases = _load_harness_cases(selection)
    by_id = {(case.task, case.doc_id): case for case in cases}
    rows = [json.loads(line) for line in inputs["rows.jsonl"].splitlines()]
    for row in rows:
        case = by_id[(row["task"], row["doc_id"])]
        if row["id"] != case.id:
            raise ValueError("saved evaluation question identity differs from pinned selection")
    header = {
        **original, "scoring_protocol": SCORING_PROTOCOL,
        "scope": "Saved-output rescoring only. Original prompts, outputs and timings are retained; no inference.",
        "operation": "saved-output-rescoring",
        "source_run": {"path": str(source_dir), "scoring_protocol": original["scoring_protocol"],
                       **{name.replace(".jsonl", "").replace(".json", "") + "_sha256": hashlib.sha256(data).hexdigest()
                          for name, data in inputs.items()}},
        "scorer_source_sha256": hashlib.sha256(Path(__file__).with_name("benchmark.py").read_bytes()).hexdigest(),
        "evaluation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run.json").open("x") as stream:
        json.dump(header, stream, indent=2, default=_json_value)
    with (output_dir / "selection.json").open("x") as stream:
        json.dump(selection, stream, indent=2)
    with (output_dir / "rows.jsonl").open("x") as stream:
        for row in rows:
            if row["outcome"] != "not_attempted":
                row = _evaluation_row(by_id[(row["task"], row["doc_id"])], row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        _record_not_attempted(stream, cases, {(row["task"], row["doc_id"]) for row in rows})
    return write_dataset_evaluation_report(output_dir)


def run_harness_evaluation(
    *, model: Any, selection: dict[str, dict[str, Any]], output_dir: str | Path,
    metadata: dict[str, Any], apply_chat_template: bool = False,
) -> dict[str, Any]:
    """Evaluate a caller-owned Engine or an existing lm-eval LM on frozen rows.

    lm-eval owns dataset loading, request construction, filters and metrics.
    This wrapper only measures requests and writes compact rows/report. It
    never installs a reference library, creates a model or executes inference
    on import. Complete question documents are not copied into result files.
    """
    _require_harness()
    from lm_eval.api.instance import Instance
    from lm_eval.api.model import LM
    from lm_eval.evaluator import evaluate

    is_engine = isinstance(model, Engine)
    if is_engine and apply_chat_template:
        raise ValueError("Engine applies its chat template; do not also apply it in the harness")
    if not is_engine and (not isinstance(model, LM) or model.world_size != 1):
        raise TypeError("Pass a public Engine or an existing single-process lm-eval LM")
    # The pinned TaskManager pops inline task names before constructing custom
    # tasks. Construct its task objects directly so request metadata and the
    # retained definitions keep the same names.
    specs, tasks, cases = _load_harness_cases(selection)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    header = {**metadata, "scoring_protocol": SCORING_PROTOCOL,
              "harness": f"{HARNESS_SOURCE}@{HARNESS_COMMIT}",
              "harness_version": HARNESS_VERSION, "selection": selection,
              "selected_questions": [
                  {"task": case.task, "doc_id": case.doc_id, "id": case.id}
                  for case in cases
              ],
              "apply_chat_template": apply_chat_template,
              "task_definitions": specs,
              "evaluation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scorer_source_sha256": hashlib.sha256(Path(__file__).with_name("benchmark.py").read_bytes()).hexdigest()}
    with (output_dir / "run.json").open("x") as stream:
        json.dump(header, stream, indent=2, default=_json_value)
    with (output_dir / "selection.json").open("x") as stream:
        json.dump(selection, stream, indent=2)
    stream = (output_dir / "rows.jsonl").open("x")
    rows = []

    class RecordedLM(LM):
        def loglikelihood(self, requests):
            raise NotImplementedError("These tasks use generation, not candidate likelihood")

        def loglikelihood_rolling(self, requests):
            raise NotImplementedError("These tasks use generation, not candidate likelihood")

        @property
        def tokenizer_name(self):
            return model.tokenizer_name

        def chat_template(self, chat_template=False):
            return model.chat_template(chat_template)

        def apply_chat_template(self, chat_history, add_generation_prompt=True):
            return model.apply_chat_template(chat_history, add_generation_prompt=add_generation_prompt)

        def generate_until(self, requests):
            responses = []
            for request in requests:
                prompt, kwargs = request.args
                if is_engine:
                    if kwargs.get("do_sample", False) or kwargs.get("temperature", 0.0) != 0.0:
                        raise ValueError("Configure sampling in the executable plan, not harness overrides")
                    measured = _measure_engine_request(model, prompt, int(kwargs["max_gen_toks"]))
                else:
                    started = time.perf_counter()
                    error = None
                    interrupted = False
                    text = ""
                    try:
                        text = model.generate_until([request])[0]
                    except (Exception, KeyboardInterrupt) as caught:
                        error = f"{type(caught).__name__}: {caught}"
                        interrupted = isinstance(caught, KeyboardInterrupt)
                    measured = {"output": text, "generation_error": error, "interrupted": interrupted,
                                "latency_ms": (time.perf_counter() - started) * 1000.0,
                                "finish_reason": None, "truncated": None, "remaining_masks": None,
                                "forwards": None, "output_tokens": None, "ttft_ms": None,
                                "runtime_latency_ms": None, "effective_tokens_per_second": None,
                                "mlx_peak_active_bytes": None, "mlx_active_bytes": None,
                                "mlx_cache_bytes": None, "decode_settings": None,
                                "process_peak_rss_bytes": _process_peak_rss_bytes()}
                task = tasks[request.task_name]
                index = selection[request.task_name]["doc_ids"][request.doc_id]
                case = _case(request.task_name, request.doc, index, prompt)
                failed = measured["generation_error"] is not None or bool(measured["remaining_masks"])
                response = "" if failed else _stop_at(measured["output"], kwargs.get("until", []))
                isolated = Instance("generate_until", request.doc, request.args, 0)
                isolated.resps = [response]
                for pipeline in task._filters:
                    pipeline.apply([isolated])
                filtered = next(iter(isolated.filtered_resps.values()))
                scores = task.process_results(request.doc, [filtered])
                parsed = ((filtered or None) if case.answer_kind == "exact"
                          else score_evaluation_answer(case, filtered)[0])
                correct = bool(scores.get("exact_match", scores.get("acc", 0)))
                row = _evaluation_row(case, measured, score=(parsed, correct))
                row["evaluated_output"] = response
                row["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                rows.append(row)
                if measured["interrupted"]:
                    raise KeyboardInterrupt("harness request interrupted")
                responses.append(response)
            return responses

    primary_error: BaseException | None = None
    try:
        result = evaluate(
            lm=RecordedLM(), task_dict=tasks,
            samples={name: source["doc_ids"] for name, source in selection.items()},
            bootstrap_iters=0, log_samples=False, apply_chat_template=apply_chat_template,
        )
        # The pinned harness reports the complete split as 'effective' even
        # with selected row IDs. Retain that upstream value, then expose the
        # actual request denominator alongside completion-visibility limits.
        result["harness_reported_n_samples"] = {
            name: dict(counts) for name, counts in result["n-samples"].items()
        }
        result["sample_accounting"] = {}
        for name, source in selection.items():
            group = [row for row in rows if row["task"] == name]
            result["n-samples"][name]["effective"] = len(group)
            result["sample_accounting"][name] = {
                "selected": len(source["doc_ids"]), "attempted": len(group),
                "completion_unknown": sum(not row["completion_observed"] for row in group),
            }
        with (output_dir / "harness-result.json").open("x") as report:
            json.dump(result, report, indent=2, default=_json_value)
        return result
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if primary_error is not None:
            _close_preserving_primary(
                lambda: _record_not_attempted(stream, cases, {(row["task"], row["doc_id"]) for row in rows}),
                primary_error, label="partial harness ledger",
            )
        stream.close()
        if primary_error is None:
            write_dataset_evaluation_report(output_dir, cases=cases)
        else:
            _close_preserving_primary(
                lambda: write_dataset_evaluation_report(output_dir, cases=cases),
                primary_error, label="harness report finalization",
            )
