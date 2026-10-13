"""Offline coverage of the docs/cookbook/studio model ID check."""

import importlib.util
import io
import json
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/check_doc_model_ids.py"
SPEC = importlib.util.spec_from_file_location("check_doc_model_ids", SCRIPT)
assert SPEC and SPEC.loader
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def row(identifier: str, model_type: str = "chat", **model: Any) -> dict[str, Any]:
    return {
        "id": identifier,
        "served_names": [identifier],
        "model": {"id": identifier, "model_type": model_type, **model},
    }


REGISTRY = {
    "schema_version": 1,
    "models": [
        row("nvidia/nemotron-3-super-120b-a12b"),
        row("meta/llama-3.3-70b-instruct"),
        row("meta/llama3-8b-instruct", deprecated=True, aliases=["ai-llama3-8b"]),
        row("NV-Embed-QA", "embedding", aliases=["ai-embed-qa-4"]),
        row("nvidia/nemotron-3-embed-1b", "embedding"),
    ],
}


def write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def notebook(*cells: tuple[str, str]) -> str:
    return json.dumps(
        {
            "cells": [
                {
                    "cell_type": kind,
                    "source": source.splitlines(keepends=True),
                    "outputs": [{"text": ["meta/llama3-8b-instruct"]}],
                }
                for kind, source in cells
            ]
        }
    )


def entries(path: Path, *items: dict[str, str]) -> Path:
    path.write_text(json.dumps({"entries": list(items)}), encoding="utf-8")
    return path


def run(
    root: Path,
    *,
    allowlist: list[dict[str, str]] | None = None,
    baseline: list[dict[str, str]] | None = None,
    registry: Path | None = None,
) -> tuple[int, str]:
    if registry is None:
        registry = root / "registry.json"
        registry.write_text(json.dumps(REGISTRY), encoding="utf-8")
    out = io.StringIO()
    status = checker.check(
        root,
        registry,
        entries(root / "allow.json", *(allowlist or [])),
        entries(root / "base.json", *(baseline or [])),
        out=out,
    )
    return status, out.getvalue()


CLEAN_DOC = """
Use `nvidia/nemotron-3-super-120b-a12b` or
ChatNVIDIA(model="meta/llama-3.3-70b-instruct").
See https://build.nvidia.com/meta/llama3-8b-instruct/modelcard and
www.example.com/meta/llama3-8b-instruct or ![img](data:image/png;base64,meta/llama2).
Clone langchain-ai/langchain-nvidia, then `pip install langchain-nvidia-ai-endpoints`.
docker run nvcr.io/nim/meta/llama3-8b-instruct:latest; read docs/chat.md and
libs/ai-endpoints/README.md. Prose like phi2 or the ai-llama3-8b word is ignored.
"""


def test_current_ids_and_non_model_slashes_pass(tmp_path: Path) -> None:
    write(tmp_path, "README.md", CLEAN_DOC)
    write(tmp_path, "libs/pkg/docs/guide.md", CLEAN_DOC)
    write(
        tmp_path,
        "studio/app.py",
        'NVIDIAEmbeddings(model="nvidia/nemotron-3-embed-1b")\n',
    )
    write(tmp_path, "cookbook/nb.ipynb", notebook(("markdown", CLEAN_DOC)))

    status, output = run(tmp_path)

    assert status == 0, output
    assert "are current" in output


@pytest.mark.parametrize(
    "rel",
    [
        "README.md",
        "libs/pkg/README.md",
        "libs/pkg/docs/guide.md",
        "libs/pkg/docs/sub/x.md",
        "studio/app.py",
    ],
)
def test_injected_deprecated_id_fails_with_file_and_line(
    tmp_path: Path, rel: str
) -> None:
    write(tmp_path, rel, 'intro\nllm = ChatNVIDIA(model="meta/llama3-8b-instruct")\n')

    status, output = run(tmp_path)

    assert status == 1
    assert f"{rel}:2: meta/llama3-8b-instruct: deprecated" in output
    assert "nvidia/nemotron-3-super-120b-a12b" in output


def test_missing_legacy_and_alias_ids_fail(tmp_path: Path) -> None:
    write(
        tmp_path,
        "README.md",
        "\n".join(
            [
                "Models such as `nvidia/neva-22b`.",
                'ChatNVIDIA(model="acme/unknown-model")',
                "python -m langchain_nvidia_ai_endpoints doctor --model other/model-x",
                "NVIDIAEmbeddings(model='NV-Embed-QA')",
                'ChatNVIDIA(model="ai-llama3-8b")',
            ]
        ),
    )

    status, output = run(tmp_path)

    assert status == 1
    assert "README.md:1: nvidia/neva-22b: not in model_registry.json" in output
    assert "README.md:2: acme/unknown-model: not in model_registry.json" in output
    assert "README.md:3: other/model-x: not in model_registry.json" in output
    assert "README.md:4: NV-Embed-QA: legacy model ID" in output
    assert "nvidia/nemotron-3-embed-1b" in output
    assert (
        "README.md:5: ai-llama3-8b: legacy alias of meta/llama3-8b-instruct" in output
    )


def test_notebook_code_and_markdown_cells_are_scanned(tmp_path: Path) -> None:
    write(
        tmp_path,
        "cookbook/sub/nb.ipynb",
        notebook(
            ("markdown", "# Title\nUses `meta/llama3-8b-instruct`.\n"),
            ("raw", "meta/llama3-8b-instruct"),
            ("code", "x = 1\ny = 2\nChatNVIDIA(model='nvidia/neva-22b')\n"),
        ),
    )

    status, output = run(tmp_path)

    assert status == 1
    assert "cookbook/sub/nb.ipynb:cell 0 line 2: meta/llama3-8b-instruct" in output
    assert "cookbook/sub/nb.ipynb:cell 2 line 3: nvidia/neva-22b" in output
    assert "cell 1" not in output
    assert output.count("meta/llama3-8b-instruct: deprecated") == 1


def test_files_outside_scan_scope_are_ignored(tmp_path: Path) -> None:
    for rel in ("docs/guide.md", "libs/pkg/src/code.py", "studio/sub/x.py", "notes.md"):
        write(tmp_path, rel, "meta/llama3-8b-instruct\n")
    write(tmp_path, ".git/README.md", "meta/llama3-8b-instruct\n")

    assert run(tmp_path)[0] == 0


def test_allowlist_and_baseline_suppress_and_report_unused(tmp_path: Path) -> None:
    write(tmp_path, "libs/pkg/docs/migration.md", "| meta/llama3-8b-instruct | new |\n")
    write(tmp_path, "cookbook/nb.ipynb", notebook(("code", "NV-Embed-QA")))

    status, output = run(
        tmp_path,
        allowlist=[
            {
                "path": "libs/*/docs/migration.md",
                "id": "meta/llama3-8b-instruct",
                "reason": "Migration table",
            },
            {"path": "README.md", "id": "nvidia/old", "reason": "Gone"},
        ],
        baseline=[{"path": "cookbook/nb.ipynb", "id": "NV-Embed-QA"}],
    )

    assert status == 0, output
    assert "note: unused allowlist entry README.md nvidia/old" in output

    status, output = run(tmp_path, allowlist=[], baseline=[])
    assert status == 1
    assert "libs/pkg/docs/migration.md:1" in output
    assert "cookbook/nb.ipynb:cell 0 line 1: NV-Embed-QA" in output


def test_allowlist_entry_requires_reason(tmp_path: Path) -> None:
    write(tmp_path, "README.md", "ok\n")

    status, _ = run(tmp_path, allowlist=[{"path": "README.md", "id": "nvidia/x"}])

    assert status == 2


def test_repository_config_is_valid() -> None:
    registry = checker.load_registry()
    for model_id in checker.RECOMMENDED.values():
        assert registry.is_current(model_id), model_id
    allowlist = checker.load_entries(checker.ALLOWLIST, need_reason=True)
    assert all(entry["reason"].strip() for entry in allowlist)
    baseline = checker.load_entries(checker.BASELINE, need_reason=False)
    assert all(e["path"].startswith("cookbook/") for e in baseline)


def test_deprecated_id_fails_with_repository_registry_and_config(
    tmp_path: Path,
) -> None:
    write(
        tmp_path,
        "libs/ai-endpoints/README.md",
        'ChatNVIDIA(model="meta/llama3-8b-instruct")\n',
    )
    out = io.StringIO()

    status = checker.check(
        tmp_path, checker.REGISTRY, checker.ALLOWLIST, checker.BASELINE, out=out
    )

    assert status == 1
    assert (
        "libs/ai-endpoints/README.md:1: meta/llama3-8b-instruct: deprecated"
        in out.getvalue()
    )


def test_main_accepts_root_argument(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write(tmp_path, "README.md", "nvidia/nemotron-3-super-120b-a12b\n")

    assert checker.main(["--root", str(tmp_path)]) == 0
    assert "are current" in capsys.readouterr().out
