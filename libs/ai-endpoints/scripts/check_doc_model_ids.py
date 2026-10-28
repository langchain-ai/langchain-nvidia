# The check deliberately reports findings on stdout/stderr.
# ruff: noqa: T201
"""Fail when docs, cookbooks or studio examples name stale NVIDIA model IDs.

Scans README.md files, ``libs/*/docs/**/*.md``, ``cookbook/**/*.ipynb`` (code and
markdown cells) and ``studio/*.py`` for model ID strings and compares them with the
reviewed registry in ``langchain_nvidia_ai_endpoints/data/model_registry.json``.

A string is treated as a model ID when it is either

* ``org/model`` where ``org`` is an organization that appears in the registry
  (``nvidia/...``, ``meta/...``), or
* any ``org/model`` passed as a quoted ``model=``/``id=``/``"model":`` value or as
  a ``--model`` argument, or
* a known bare legacy ID (``NV-Embed-QA``) or a quoted legacy registry alias.

URLs, ``data:`` URIs and path segments (``.../nim/meta/...``) are ignored, so
repository slugs such as ``langchain-ai/langchain-nvidia`` and package names do
not match. A finding is any ID that is deprecated, missing from the registry, or
legacy.

Checks are context-aware. Each mention is either *hosted* or *self-hosted*:

* a fenced Markdown code block, notebook code cell or ``studio/*.py`` file is
  self-hosted when it sets a non-hosted ``base_url``/``endpoint``/
  ``NVIDIA_BASE_URL``/``--base-url`` (for example ``http://localhost:8000/v1``)
  or names a ``nvcr.io/nim/`` container, and hosted otherwise;
* other lines are self-hosted when they contain one of those signals or the
  file is listed in ``SELF_HOSTED_DOCS``, and hosted otherwise.

Hosted mentions must use IDs that the hosted API Catalog still serves: an ID
whose registry ``provenance.hosted.state`` is ``unavailable`` is a finding.
Self-hosted mentions may use such downloadable-only IDs only when they are
explicitly allowlisted with ``"context": "self-hosted"``, and only when their
``provenance.downloadable.state`` is ``current``.

Intentional mentions (for example a migration table naming old IDs) go in
``doc_model_ids_allowlist.json`` with a reason. Known hits that another change
will remove go in ``doc_model_ids_baseline.json``. Both files hold
``{"entries": [{"path": <glob relative to the repo root>, "id": <model id>}]}``;
allowlist entries also need a non-empty ``"reason"``. An optional ``"context"``
(``"hosted"`` or ``"self-hosted"``) limits an entry to mentions in that context.

Exit status: 0 when clean, 1 when there are findings, 2 on configuration errors.
Run from any directory: ``python3 libs/ai-endpoints/scripts/check_doc_model_ids.py``.
"""

import argparse
import fnmatch
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import urlparse

SCRIPTS = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPTS.parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
REGISTRY = PACKAGE_ROOT / "langchain_nvidia_ai_endpoints/data/model_registry.json"
ALLOWLIST = SCRIPTS / "doc_model_ids_allowlist.json"
BASELINE = SCRIPTS / "doc_model_ids_baseline.json"

SCAN_GLOBS = (
    "**/README.md",
    "libs/*/docs/**/*.md",
    "cookbook/**/*.ipynb",
    "studio/*.py",
)
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", ".mypy_cache", ".tox"}

HOSTED = "hosted"
SELF_HOSTED = "self-hosted"
CONTEXTS = (HOSTED, SELF_HOSTED)
# Docs whose prose and tables describe self-hosted NIMs. Their code blocks still
# get their context from their own content.
SELF_HOSTED_DOCS = ("libs/ai-endpoints/docs/self_hosting.md",)
HOSTED_HOSTS = {"integrate.api.nvidia.com", "ai.api.nvidia.com"}

# Bare IDs that predate the ``org/model`` naming; still resolvable but not to be
# shown in examples.
LEGACY_IDS = ("NV-Embed-QA", "nv-rerank-qa-mistral-4b:1", "nv-rerank-qa-mistral-4b")

# Suggested replacements per registry model type. Only suggested while current.
RECOMMENDED = {
    "chat": "nvidia/nemotron-3-super-120b-a12b",
    "vlm": "nvidia/nemotron-nano-12b-v2-vl",
    "embedding": "nvidia/nemotron-3-embed-1b",
    "ranking": "nvidia/llama-nemotron-rerank-vl-1b-v2",
    "ranking-vlm": "nvidia/llama-nemotron-rerank-vl-1b-v2",
}

_URL_CHARS = r"""[^\s"'<>()\[\]`]+"""
_STRIP = re.compile(
    rf"[A-Za-z][\w+.-]*://{_URL_CHARS}|\bdata:[\w/+.-]+;base64,{_URL_CHARS}"
    rf"|\bwww\.{_URL_CHARS}"
)
_SLASH_ID = re.compile(
    r"(?<![\w./@:-])([A-Za-z0-9][\w-]*)/([A-Za-z0-9](?:[\w.:-]*\w)?)(?![\w/])"
)
_BASE_URL = re.compile(
    r"""(?:\b(?:base_url|endpoint)|NVIDIA_BASE_URL["']?\]?|--base-url)"""
    r"""\s*[=: ]\s*["']?(https?://[^\s"'<>(),]+)"""
)
_NIM_IMAGE = re.compile(r"\bnvcr\.io/nim/")
_FENCE = re.compile(r"^\s*(```|~~~)")
_MODEL_ARG = re.compile(
    r"""(?:\b(?:model|model_name|model_id|id)\s*[=:]\s*|["']model["']\s*:\s*)"""
    r"""["']([^"'\s]+/[^"'\s]+)["']|--model[ =]["']?([^\s"']+/[^\s"']+)"""
)


@dataclass(frozen=True)
class Finding:
    path: str
    location: str
    model_id: str
    problem: str
    context: str = HOSTED

    def __str__(self) -> str:
        return f"{self.path}:{self.location}: {self.model_id}: {self.problem}"


@dataclass(frozen=True)
class Registry:
    models: dict[str, dict[str, Any]]
    aliases: dict[str, str]
    provenance: dict[str, dict[str, Any]]

    @property
    def orgs(self) -> set[str]:
        return {i.split("/", 1)[0].lower() for i in self.models if "/" in i}

    def is_current(self, model_id: str) -> bool:
        model = self.models.get(model_id)
        return model is not None and not model.get("deprecated", False)

    def state(self, model_id: str, source: str) -> str:
        info = self.provenance.get(model_id, {}).get(source) or {}
        return str(info.get("state", "unknown"))

    def is_hosted_current(self, model_id: str) -> bool:
        return self.is_current(model_id) and self.state(model_id, "hosted") == "current"

    def suggestion(self, model_type: str | None, *, hosted: bool = False) -> str:
        candidate = RECOMMENDED.get(model_type or "")
        current = self.is_hosted_current if hosted else self.is_current
        kind = "hosted-current" if hosted else "current"
        if candidate and current(candidate):
            return f"; use a {kind} registry ID such as {candidate}"
        return f"; use a {kind} registry ID of the same model type"


def load_registry(path: Path = REGISTRY) -> Registry:
    data = json.loads(path.read_text(encoding="utf-8"))
    models: dict[str, dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for row in data["models"]:
        model = row["model"]
        for name in [row["id"], *row.get("served_names", [])]:
            models[name] = model
            provenance[name] = row.get("provenance") or {}
        for alias in model.get("aliases") or []:
            aliases[alias] = row["id"]
    return Registry(models, aliases, provenance)


def load_entries(path: Path, *, need_reason: bool) -> list[dict[str, str]]:
    if not path.exists():
        return []
    entries = json.loads(path.read_text(encoding="utf-8")).get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected an 'entries' list")
    for entry in entries:
        if not isinstance(entry, dict) or not all(
            isinstance(entry.get(key), str) and entry[key] for key in ("path", "id")
        ):
            raise ValueError(f"{path}: every entry needs 'path' and 'id': {entry}")
        if need_reason and not str(entry.get("reason", "")).strip():
            raise ValueError(f"{path}: allowlist entry needs a reason: {entry}")
        if entry.get("context", HOSTED) not in CONTEXTS:
            raise ValueError(f"{path}: context must be one of {CONTEXTS}: {entry}")
    return entries


def iter_files(root: Path) -> Iterator[Path]:
    seen: set[Path] = set()
    for pattern in SCAN_GLOBS:
        for path in sorted(root.glob(pattern)):
            rel = path.relative_to(root)
            if path in seen or not path.is_file() or SKIP_DIRS & set(rel.parts):
                continue
            seen.add(path)
            yield path


def is_self_hosted(text: str) -> bool:
    """Whether text points at a non-hosted endpoint or a NIM container."""
    for match in _BASE_URL.finditer(text):
        if urlparse(match.group(1)).hostname not in HOSTED_HOSTS:
            return True
    return bool(_NIM_IMAGE.search(text))


def _block_context(lines: list[str]) -> str:
    return SELF_HOSTED if is_self_hosted("\n".join(lines)) else HOSTED


def _markdown_contexts(lines: list[str], prose_default: str) -> list[str]:
    """Context per line: fenced blocks as a whole, other lines one by one."""
    contexts: list[str] = []
    index = 0
    while index < len(lines):
        if _FENCE.match(lines[index]):
            end = index + 1
            while end < len(lines) and not _FENCE.match(lines[end]):
                end += 1
            block = lines[index : end + 1]
            contexts.extend([_block_context(block)] * len(block))
            index = end + 1
            continue
        line = lines[index]
        contexts.append(SELF_HOSTED if is_self_hosted(line) else prose_default)
        index += 1
    return contexts


def iter_lines(path: Path, rel: str = "") -> Iterator[tuple[str, str, str]]:
    """Yield ``(location, text, context)``; notebooks use ``cell N line M``."""
    text = path.read_text(encoding="utf-8")
    prose_default = SELF_HOSTED if rel in SELF_HOSTED_DOCS else HOSTED
    if path.suffix != ".ipynb":
        lines = text.splitlines()
        if path.suffix == ".py":
            contexts = [_block_context(lines)] * len(lines)
        else:
            contexts = _markdown_contexts(lines, prose_default)
        for number, (line, context) in enumerate(zip(lines, contexts), 1):
            yield str(number), line, context
        return
    for index, cell in enumerate(json.loads(text).get("cells", [])):
        kind = cell.get("cell_type")
        if kind not in ("code", "markdown"):
            continue
        source = cell.get("source", "")
        lines = "".join(source).splitlines() if isinstance(source, list) else []
        if isinstance(source, str):
            lines = source.splitlines()
        if kind == "code":
            contexts = [_block_context(lines)] * len(lines)
        else:
            contexts = _markdown_contexts(lines, prose_default)
        for number, (line, context) in enumerate(zip(lines, contexts), 1):
            yield f"cell {index} line {number}", line, context


def scan_line(
    line: str, registry: Registry, context: str = HOSTED
) -> Iterator[tuple[str, str]]:
    """Yield ``(model_id, problem)`` for every stale model ID in a line."""
    line = _STRIP.sub(" ", line)
    candidates = {
        f"{org}/{name}"
        for org, name in _SLASH_ID.findall(line)
        if org.lower() in registry.orgs
    }
    candidates |= {a or b for a, b in _MODEL_ARG.findall(line)}
    for model_id in sorted(candidates):
        model = registry.models.get(model_id)
        if model is None:
            yield model_id, "not in model_registry.json" + registry.suggestion(None)
        elif model.get("deprecated", False):
            yield (
                model_id,
                "deprecated in model_registry.json"
                + registry.suggestion(model.get("model_type")),
            )
        elif registry.state(model_id, "hosted") == "unavailable":
            model_type = model.get("model_type")
            if context == HOSTED:
                yield (
                    model_id,
                    "retired from hosted endpoints (provenance.hosted.state is "
                    "unavailable) in a hosted example"
                    + registry.suggestion(model_type, hosted=True)
                    + ", or point the example at a self-hosted NIM base_url",
                )
            elif registry.state(model_id, "downloadable") == "current":
                yield (
                    model_id,
                    "downloadable-only ID in a self-hosted example; allowlist it "
                    'with "context": "self-hosted"',
                )
            else:
                yield (
                    model_id,
                    "neither hosted nor a current downloadable NIM"
                    + registry.suggestion(model_type),
                )
    for legacy in LEGACY_IDS:
        if re.search(rf"(?<![\w/:-]){re.escape(legacy)}(?![\w:-])", line):
            model_type = (registry.models.get(legacy) or {}).get("model_type")
            yield legacy, "legacy model ID" + registry.suggestion(model_type)
    for alias, target in registry.aliases.items():
        if alias in LEGACY_IDS or not re.search(
            rf"""["']{re.escape(alias)}["']""", line
        ):
            continue
        model_type = registry.models[target].get("model_type")
        yield alias, f"legacy alias of {target}" + registry.suggestion(model_type)


def scan(root: Path, registry: Registry) -> list[Finding]:
    findings = []
    for path in iter_files(root):
        rel = path.relative_to(root).as_posix()
        for location, line, context in iter_lines(path, rel):
            for model_id, problem in scan_line(line, registry, context):
                findings.append(Finding(rel, location, model_id, problem, context))
    return findings


def _matches(finding: Finding, entry: dict[str, str]) -> bool:
    return (
        entry["id"] == finding.model_id
        and entry.get("context", finding.context) == finding.context
        and fnmatch.fnmatchcase(finding.path, entry["path"])
    )


def check_self_hosted_entries(
    registry: Registry, allowlist: list[dict[str, str]]
) -> None:
    """Self-hosted allowlist entries must name current downloadable NIMs."""
    for entry in allowlist:
        if entry.get("context") != SELF_HOSTED:
            continue
        if registry.state(entry["id"], "downloadable") != "current":
            raise ValueError(
                "self-hosted allowlist entry needs provenance.downloadable.state "
                f"current in model_registry.json: {entry}"
            )


def check(
    root: Path,
    registry_path: Path = REGISTRY,
    allowlist_path: Path = ALLOWLIST,
    baseline_path: Path = BASELINE,
    out: Any = None,
) -> int:
    out = out or sys.stdout
    try:
        registry = load_registry(registry_path)
        allowlist = load_entries(allowlist_path, need_reason=True)
        baseline = load_entries(baseline_path, need_reason=False)
        check_self_hosted_entries(registry, allowlist)
    except (OSError, ValueError, KeyError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    used: set[int] = set()
    failures = []
    for finding in scan(root, registry):
        hits = [
            i
            for i, entry in enumerate([*allowlist, *baseline])
            if _matches(finding, entry)
        ]
        used.update(hits)
        if not hits:
            failures.append(finding)
    for i, entry in enumerate([*allowlist, *baseline]):
        if i not in used:
            source = "allowlist" if i < len(allowlist) else "baseline"
            print(
                f"note: unused {source} entry {entry['path']} {entry['id']}; "
                "remove it",
                file=out,
            )
    for finding in failures:
        print(finding, file=out)
    if failures:
        print(
            f"{len(failures)} stale model ID mention(s). Replace them with current "
            "IDs from model_registry.json, or allowlist intentional mentions in "
            f"{allowlist_path.name}.",
            file=out,
        )
        return 1
    print("Model IDs in docs, cookbooks and studio are current", file=out)
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--allowlist", type=Path, default=ALLOWLIST)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    args = parser.parse_args(None if argv is None else list(argv))
    return check(args.root, args.registry, args.allowlist, args.baseline)


if __name__ == "__main__":
    sys.exit(main())
