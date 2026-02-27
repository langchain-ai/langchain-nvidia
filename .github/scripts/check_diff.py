import json
import sys
import os
from typing import Dict

NVIDIA_DIRS = [
    "libs/ai-endpoints",
    "libs/langgraph",
    "libs/trt",
]

PYTHON_VERSIONS = {
    "libs/ai-endpoints": ["3.10", "3.11", "3.12", "3.13"],
    "libs/langgraph": ["3.11", "3.12", "3.13"],
    "libs/trt": ["3.9", "3.10", "3.11"],
}

LINT_PYTHON_VERSIONS = {
    "libs/ai-endpoints": ["3.10", "3.13"],
    "libs/langgraph": ["3.11", "3.13"],
    "libs/trt": ["3.9", "3.11"],
}

if __name__ == "__main__":
    files = sys.argv[1:]

    dirs_to_run: Dict[str, set] = {
        "lint": set(),
        "test": set(),
    }

    if len(files) == 300:
        # max diff length is 300 files - there are likely files missing
        raise ValueError("Max diff reached. Please manually run CI on changed libs.")

    for file in files:
        if any(
            file.startswith(dir_)
            for dir_ in (
                ".github/workflows",
                ".github/tools",
                ".github/actions",
                ".github/scripts/check_diff.py",
            )
        ):
            # add all LANGCHAIN_DIRS for infra changes
            # dirs_to_run["lint"].add(".")
            pass

        if any(file.startswith(dir_) for dir_ in NVIDIA_DIRS):
            for dir_ in NVIDIA_DIRS:
                if file.startswith(dir_):
                    # add that dir and all dirs after in LANGCHAIN_DIRS
                    # for extended testing
                    dirs_to_run["test"].add(dir_)
        elif file.startswith("libs/"):
            raise ValueError(
                f"Unknown lib: {file}. check_diff.py likely needs "
                "an update for this new library!"
            )
        # elif any(file.startswith(p) for p in ["docs/", "templates/", "cookbook/"]):
        #     dirs_to_run["lint"].add(".")

    test_matrix = [
        {
            "working-directory": dir_,
            "python-versions": json.dumps(PYTHON_VERSIONS.get(dir_, ["3.8", "3.9", "3.10", "3.11"]))
        }
        for dir_ in dirs_to_run["test"]
    ]
    
    lint_matrix = [
        {
            "working-directory": dir_,
            "python-versions": json.dumps(LINT_PYTHON_VERSIONS.get(dir_, ["3.8", "3.11"]))
        }
        for dir_ in (dirs_to_run["lint"] | dirs_to_run["test"])
    ]
    
    outputs = {
        "dirs-to-lint": list(
            dirs_to_run["lint"] | dirs_to_run["test"]
        ),
        "dirs-to-test": list(dirs_to_run["test"]),
        "test-matrix": test_matrix,
        "lint-matrix": lint_matrix,
    }
    for key, value in outputs.items():
        json_output = json.dumps(value)
        print(f"{key}={json_output}")  # noqa: T201
