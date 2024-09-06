from typing import Any, Dict, List, Union

import pytest

from langchain_nvidia_ai_endpoints.chat_models import _nv_vlm_get_asset_ids


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
