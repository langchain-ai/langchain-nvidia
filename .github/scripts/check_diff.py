import json
import sys
import os
from typing import Dict

NVIDIA_DIRS = [
    "libs/ai-endpoints",
    "libs/trt",
]

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
            dirs_to_run["extended-test"].update(NVIDIA_DIRS)
            dirs_to_run["lint"].add(".")

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
        elif any(file.startswith(p) for p in ["docs/", "templates/", "cookbook/"]):
            dirs_to_run["lint"].add(".")

    outputs = {
        "dirs-to-lint": list(
            dirs_to_run["lint"] | dirs_to_run["test"] | dirs_to_run["extended-test"]
        ),
        "dirs-to-test": list(dirs_to_run["test"] | dirs_to_run["extended-test"]),
    }
    for key, value in outputs.items():
        json_output = json.dumps(value)
        print(f"{key}={json_output}")  # noqa: T201
