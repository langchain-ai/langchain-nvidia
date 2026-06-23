from typing import Any, Dict, List, Union

import pytest

from langchain_nvidia_ai_endpoints._utils import _url_to_b64_string
from langchain_nvidia_ai_endpoints.chat_models import _nv_vlm_get_asset_ids


def test_url_to_b64_string_rejects_local_file_path(
    monkeypatch: Any, tmp_path: Any
) -> None:
    local_file = tmp_path / "secret.txt"
    local_file.write_text("TOP-SECRET", encoding="utf-8")

    def fail_open(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("local files must not be opened")

    monkeypatch.setattr("builtins.open", fail_open)

    with pytest.raises(ValueError, match="Local file paths are no longer supported"):
        _url_to_b64_string(str(local_file))


@pytest.mark.parametrize(
    "content, expected",
    [
        # Single asset ID in a string (double quotes)
        ('<img src="data:image/png;asset_id,12345"/>', ["12345"]),
        # Multiple asset IDs in a string (double quotes)
        (
            (
                '<img src="data:image/png;asset_id,12345"/>'
                '<img src="data:image/jpeg;asset_id,67890"/>'
            ),
            ["12345", "67890"],
        ),
        # Single asset ID in list of strings (single quotes)
        (["<img src='data:image/png;asset_id,12345'/>"], ["12345"]),
        # Multiple asset IDs in list of strings (single quotes)
        (
            [
                "<img src='data:image/png;asset_id,12345'/>",
                "<img src='data:image/jpeg;asset_id,67890'/>",
            ],
            ["12345", "67890"],
        ),
        # Single asset ID in a list of dictionaries
        ([{"image_url": {"url": "data:image/png;asset_id,12345"}}], ["12345"]),
        # Multiple asset IDs in a list of dictionaries
        (
            [
                {"image_url": {"url": "data:image/png;asset_id,12345"}},
                {"image_url": {"url": "data:image/jpeg;asset_id,67890"}},
            ],
            ["12345", "67890"],
        ),
        # No asset IDs present (double quotes)
        ('<img src="data:image/png;no_asset_id"/>', []),
        # No asset IDs present (single quotes)
        ("<img src='data:image/png;no_asset_id'/>", []),
    ],
    ids=[
        "single_asset_id_string_double_quotes",
        "multiple_asset_ids_string_double_quotes",
        "single_asset_id_list_of_strings_single_quotes",
        "multiple_asset_ids_list_of_strings_single_quotes",
        "single_asset_id_list_of_dicts",
        "multiple_asset_ids_list_of_dicts",
        "no_asset_ids_double_quotes",
        "no_asset_ids_single_quotes",
    ],
)
def test_nv_vlm_get_asset_ids(
    content: Union[str, List[Union[str, Dict[str, Any]]]], expected: List[str]
) -> None:
    result = _nv_vlm_get_asset_ids(content)
    assert result == expected
