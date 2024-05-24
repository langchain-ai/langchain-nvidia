import inspect

import pytest

import langchain_nvidia_ai_endpoints


@pytest.fixture(
    params=[
        member[1]
        for member in inspect.getmembers(langchain_nvidia_ai_endpoints, inspect.isclass)
    ]
)
def public_class(request: pytest.FixtureRequest) -> type:
    return request.param
