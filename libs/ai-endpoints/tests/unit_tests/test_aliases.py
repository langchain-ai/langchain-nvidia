import pytest

from langchain_nvidia_ai_endpoints._statics import MODEL_TABLE


@pytest.mark.parametrize(
    "alias",
    [
        alias
        for model in MODEL_TABLE.values()
        if model.aliases is not None
        for alias in model.aliases
    ],
)
def test_aliases(public_class: type, alias: str) -> None:
    with pytest.warns(UserWarning):
        x = public_class(model=alias)
        assert x.model == x._client.model
