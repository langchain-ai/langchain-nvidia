import base64
from typing import Any, Dict, List, Union

import pytest
from langchain_core.messages import BaseMessage, HumanMessage

from langchain_nvidia_ai_endpoints.chat_models import ChatNVIDIA

# todo: test S3 bucket asset id
# todo: sizes
# todo: formats
# todo: multiple images
# todo: multiple texts
# todo: detail (fidelity)


#
# note: differences between api catalog and openai api
#  - openai api supports server-side image download, api catalog does not
#   - ChatNVIDIA does client side download to simulate the same behavior
#  - ChatNVIDIA will automatically read local files and convert them to base64
#  - openai api uses {"image_url": {"url": "..."}}
#     where api catalog uses {"image_url": "..."}
#


@pytest.mark.parametrize(
    "content",
    [
        [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d7/Boston_-_panoramio_(23).jpg/2560px-Boston_-_panoramio_(23).jpg"
                },
            }
        ],
        [{"type": "image_url", "image_url": {"url": "tests/data/nvidia-picasso.jpg"}}],
        [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"""data:image/png;base64,{
                        base64.b64encode(
                            open('tests/data/nvidia-picasso.jpg', 'rb').read()
                        ).decode('utf-8')
                    }"""
                },
            }
        ],
    ],
    ids=["url", "file", "tag"],
)
def test_vlm_model(
    vlm_model: str, mode: dict, content: Union[str, List[Union[str, Dict[Any, Any]]]]
) -> None:
    chat = ChatNVIDIA(model=vlm_model, **mode)
    response = chat.invoke([HumanMessage(content=content)])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

    for token in chat.stream([HumanMessage(content=content)]):
        assert isinstance(token.content, str)
