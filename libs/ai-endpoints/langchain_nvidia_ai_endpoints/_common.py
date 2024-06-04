from __future__ import annotations

import json
import logging
import os
import time
import warnings
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from urllib.parse import urlparse

import aiohttp
import requests
from langchain_core.pydantic_v1 import (
    BaseModel,
    Field,
    PrivateAttr,
    SecretStr,
    root_validator,
    validator,
)
from requests.models import Response

from langchain_nvidia_ai_endpoints._statics import MODEL_TABLE, Model, determine_model

logger = logging.getLogger(__name__)

_MODE_TYPE = Literal["nvidia", "nim"]


def default_payload_fn(payload: dict) -> dict:
    return payload


class NVEModel(BaseModel):

    """
    Underlying Client for interacting with the AI Foundation Model Function API.
    Leveraged by the NVIDIABaseModel to provide a simple requests-oriented interface.
    Direct abstraction over NGC-recommended streaming/non-streaming Python solutions.

    NOTE: Models in the playground does not currently support raw text continuation.
    """

    # todo: add a validator for requests.Response (last_response attribute) and
    #       remove arbitrary_types_allowed=True
    class Config:
        arbitrary_types_allowed = True

    ## Core defaults. These probably should not be changed
    _api_key_var = "NVIDIA_API_KEY"
    base_url: str = Field(
        ...,
        description="Base URL for standard inference",
    )
    infer_path: str = Field(
        ...,
        description="Path for inference",
    )
    listing_path: str = Field(
        "{base_url}/models",
        description="Path for listing available models",
    )
    polling_endpoint: str = Field(
        "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/{request_id}",
        description="Path for polling after HTTP 202 responses",
    )
    get_session_fn: Callable = Field(requests.Session)
    get_asession_fn: Callable = Field(aiohttp.ClientSession)

    api_key: Optional[SecretStr] = Field(description="API Key for service of choice")

    ## Generation arguments
    timeout: float = Field(60, ge=0, description="Timeout for waiting on response (s)")
    interval: float = Field(0.02, ge=0, description="Interval for pulling response")
    last_inputs: dict = Field({}, description="Last inputs sent over to the server")
    last_response: Response = Field(
        None, description="Last response sent from the server"
    )
    payload_fn: Callable = Field(
        default_payload_fn, description="Function to process payload"
    )
    headers_tmpl: dict = Field(
        {
            "call": {
                "Accept": "application/json",
                "Authorization": "Bearer {api_key}",
                "User-Agent": "langchain-nvidia-ai-endpoints",
            },
            "stream": {
                "Accept": "text/event-stream",
                "Content-Type": "application/json",
                "Authorization": "Bearer {api_key}",
                "User-Agent": "langchain-nvidia-ai-endpoints",
            },
        },
        description="Headers template must contain `call` and `stream` keys.",
    )
    _available_models: Optional[List[Model]] = PrivateAttr(default=None)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"api_key": self._api_key_var}

    @property
    def headers(self) -> dict:
        """Return headers with API key injected"""
        headers_ = self.headers_tmpl.copy()
        for header in headers_.values():
            if "{api_key}" in header["Authorization"] and self.api_key:
                header["Authorization"] = header["Authorization"].format(
                    api_key=self.api_key.get_secret_value(),
                )
        return headers_

    @validator("base_url")
    def _validate_base_url(cls, v: str) -> str:
        if v is not None:
            result = urlparse(v)
            # Ensure scheme and netloc (domain name) are present
            if not (result.scheme and result.netloc):
                raise ValueError(
                    f"Invalid base_url, minimally needs scheme and netloc: {v}"
                )
        return v

    @root_validator(pre=True)
    def _validate_model(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and update model arguments, including API key and formatting"""
        values["api_key"] = (
            values.get(cls._api_key_var.lower())
            or values.get("api_key")
            or os.getenv(cls._api_key_var)
            or None
        )
        return values

    @property
    def available_models(self) -> list[Model]:
        """List the available models that can be invoked."""
        if self._available_models is not None:
            return self._available_models

        response, _ = self._get(self.listing_path.format(base_url=self.base_url))
        # expecting -
        # {"object": "list",
        #  "data": [
        #   {
        #     "id": "{name of model}",
        #     "object": "model",
        #     "created": {some int},
        #     "owned_by": "{some owner}"
        #   },
        #   ...
        #  ]
        # }
        assert response.status_code == 200, "Failed to get models"
        assert "data" in response.json(), "No data found in response"
        self._available_models = []
        for element in response.json()["data"]:
            assert "id" in element, f"No id found in {element}"
            if not (model := determine_model(element["id"])):
                # model is not in table of known models, but it exists
                # so we'll let it through. use of this model will be
                # accompanied by a warning.
                model = Model(id=element["id"])
            self._available_models.append(model)

        return self._available_models

    ####################################################################################
    ## Core utilities for posting and getting from NV Endpoints

    def _post(
        self,
        invoke_url: str,
        payload: Optional[dict] = {},
    ) -> Tuple[Response, Any]:
        """Method for posting to the AI Foundation Model Function API."""
        self.last_inputs = {
            "url": invoke_url,
            "headers": self.headers["call"],
            "json": self.payload_fn(payload),
            "stream": False,
        }
        session = self.get_session_fn()
        self.last_response = response = session.post(**self.last_inputs)
        self._try_raise(response)
        return response, session

    def _get(
        self,
        invoke_url: str,
        payload: Optional[dict] = {},
    ) -> Tuple[Response, Any]:
        """Method for getting from the AI Foundation Model Function API."""
        self.last_inputs = {
            "url": invoke_url,
            "headers": self.headers["call"],
            "stream": False,
        }
        if payload:
            self.last_inputs["json"] = self.payload_fn(payload)
        session = self.get_session_fn()
        self.last_response = response = session.get(**self.last_inputs)
        self._try_raise(response)
        return response, session

    def _wait(self, response: Response, session: Any) -> Response:
        """
        Any request may return a 202 status code, which means the request is still
        processing. This method will wait for a response using the request id.

        see https://docs.nvidia.com/cloud-functions/user-guide/latest/cloud-function/api.html#http-polling
        """
        start_time = time.time()
        # note: the local NIM does not return a 202 status code
        #       (per RL 22may2024 circa 24.05)
        while (
            response.status_code == 202
        ):  # todo: there are no tests that reach this point
            time.sleep(self.interval)
            if (time.time() - start_time) > self.timeout:
                raise TimeoutError(
                    f"Timeout reached without a successful response."
                    f"\nLast response: {str(response)}"
                )
            assert (
                "NVCF-REQID" in response.headers
            ), "Received 202 response with no request id to follow"
            request_id = response.headers.get("NVCF-REQID")
            self.last_response = response = session.get(
                self.polling_endpoint.format(request_id=request_id),
                headers=self.headers["call"],
            )
        self._try_raise(response)
        return response

    def _try_raise(self, response: Response) -> None:
        """Try to raise an error from a response"""
        try:
            response.raise_for_status()
        except requests.HTTPError:
            try:
                rd = response.json()
                if "detail" in rd and "reqId" in rd.get("detail", ""):
                    rd_buf = "- " + str(rd["detail"])
                    rd_buf = rd_buf.replace(": ", ", Error: ").replace(", ", "\n- ")
                    rd["detail"] = rd_buf
            except json.JSONDecodeError:
                rd = response.__dict__
                if "status_code" in rd:
                    if "headers" in rd and "WWW-Authenticate" in rd["headers"]:
                        rd["detail"] = rd.get("headers").get("WWW-Authenticate")
                        rd["detail"] = rd["detail"].replace(", ", "\n")
                else:
                    rd = rd.get("_content", rd)
                    if isinstance(rd, bytes):
                        rd = rd.decode("utf-8")[5:]  ## remove "data:" prefix
                    try:
                        rd = json.loads(rd)
                    except Exception:
                        rd = {"detail": rd}
            status = rd.get("status") or rd.get("status_code") or "###"
            title = (
                rd.get("title")
                or rd.get("error")
                or rd.get("reason")
                or "Unknown Error"
            )
            header = f"[{status}] {title}"
            body = ""
            if "requestId" in rd:
                if "detail" in rd:
                    body += f"{rd['detail']}\n"
                body += "RequestID: " + rd["requestId"]
            else:
                body = rd.get("detail", rd)
            if str(status) == "401":
                body += "\nPlease check or regenerate your API key."
            # todo: raise as an HTTPError
            raise Exception(f"{header}\n{body}") from None

    ####################################################################################
    ## Simple query interface to show the set of model options

    def query(
        self,
        invoke_url: str,
        payload: Optional[dict] = None,
        request: str = "get",
    ) -> dict:
        """Simple method for an end-to-end get query. Returns result dictionary"""
        if request == "get":
            response, session = self._get(invoke_url, payload)
        else:
            response, session = self._post(invoke_url, payload)
        response = self._wait(response, session)
        output = self._process_response(response)[0]
        return output

    def _process_response(self, response: Union[str, Response]) -> List[dict]:
        """General-purpose response processing for single responses and streams"""
        if hasattr(response, "json"):  ## For single response (i.e. non-streaming)
            try:
                return [response.json()]
            except json.JSONDecodeError:
                response = str(response.__dict__)
        if isinstance(response, str):  ## For set of responses (i.e. streaming)
            msg_list = []
            for msg in response.split("\n\n"):
                if "{" not in msg:
                    continue
                msg_list += [json.loads(msg[msg.find("{") :])]
            return msg_list
        raise ValueError(f"Received ill-formed response: {response}")

    def _get_invoke_url(
        self,
        invoke_url: Optional[str] = None,
    ) -> str:
        """Helper method to get invoke URL from a model name, URL, or endpoint stub"""
        if not invoke_url:
            invoke_url = self.infer_path.format(base_url=self.base_url)

        return invoke_url

    ####################################################################################
    ## Generation interface to allow users to generate new values from endpoints

    def get_req(
        self,
        payload: dict = {},
        invoke_url: Optional[str] = None,
    ) -> Response:
        """Post to the API."""
        invoke_url = self._get_invoke_url(invoke_url)
        if payload.get("stream", False) is True:
            payload = {**payload, "stream": False}
        response, session = self._post(invoke_url, payload)
        return self._wait(response, session)

    def get_req_generation(
        self,
        payload: dict = {},
        invoke_url: Optional[str] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> dict:
        """Method for an end-to-end post query with NVE post-processing."""
        invoke_url = self._get_invoke_url(invoke_url)
        response = self.get_req(payload, invoke_url)
        output, _ = self.postprocess(response, stop=stop)
        return output

    def postprocess(
        self, response: Union[str, Response], stop: Optional[Sequence[str]] = None
    ) -> Tuple[dict, bool]:
        """Parses a response from the AI Foundation Model Function API.
        Strongly assumes that the API will return a single response.
        """
        msg_list = self._process_response(response)
        msg, is_stopped = self._aggregate_msgs(msg_list)
        msg, is_stopped = self._early_stop_msg(msg, is_stopped, stop=stop)
        return msg, is_stopped

    def _aggregate_msgs(self, msg_list: Sequence[dict]) -> Tuple[dict, bool]:
        """Dig out relevant details of aggregated message"""
        content_buffer: Dict[str, Any] = dict()
        content_holder: Dict[Any, Any] = dict()
        usage_holder: Dict[Any, Any] = dict()  ####
        is_stopped = False
        for msg in msg_list:
            usage_holder = msg.get("usage", {})  ####
            if "choices" in msg:
                ## Tease out ['choices'][0]...['delta'/'message']
                msg = msg.get("choices", [{}])[0]
                is_stopped = msg.get("finish_reason", "") == "stop"
                msg = msg.get("delta", msg.get("message", msg.get("text", "")))
                if not isinstance(msg, dict):
                    msg = {"content": msg}
            elif "data" in msg:
                ## Tease out ['data'][0]...['embedding']
                msg = msg.get("data", [{}])[0]
            content_holder = msg
            for k, v in msg.items():
                if k in ("content",) and k in content_buffer:
                    content_buffer[k] += v
                else:
                    content_buffer[k] = v
            if is_stopped:
                break
        content_holder = {**content_holder, **content_buffer}
        if usage_holder:
            content_holder.update(token_usage=usage_holder)  ####
        return content_holder, is_stopped

    def _early_stop_msg(
        self, msg: dict, is_stopped: bool, stop: Optional[Sequence[str]] = None
    ) -> Tuple[dict, bool]:
        """Try to early-terminate streaming or generation by iterating over stop list"""
        content = msg.get("content", "")
        if content and stop:
            for stop_str in stop:
                if stop_str and stop_str in content:
                    msg["content"] = content[: content.find(stop_str) + 1]
                    is_stopped = True
        return msg, is_stopped

    ####################################################################################
    ## Streaming interface to allow you to iterate through progressive generations

    def get_req_stream(
        self,
        payload: dict = {},
        invoke_url: Optional[str] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> Iterator:
        invoke_url = self._get_invoke_url(invoke_url)
        if payload.get("stream", True) is False:
            payload = {**payload, "stream": True}
        self.last_inputs = {
            "url": invoke_url,
            "headers": self.headers["stream"],
            "json": self.payload_fn(payload),
            "stream": True,
        }
        response = self.get_session_fn().post(**self.last_inputs)
        self._try_raise(response)
        call = self.copy()

        def out_gen() -> Generator[dict, Any, Any]:
            ## Good for client, since it allows self.last_inputs
            for line in response.iter_lines():
                if line and line.strip() != b"data: [DONE]":
                    line = line.decode("utf-8")
                    msg, final_line = call.postprocess(line, stop=stop)
                    yield msg
                    if final_line:
                        break
                self._try_raise(response)

        return (r for r in out_gen())

    ####################################################################################
    ## Asynchronous streaming interface to allow multiple generations to happen at once.

    async def get_req_astream(
        self,
        payload: dict = {},
        invoke_url: Optional[str] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> AsyncIterator:
        invoke_url = self._get_invoke_url(invoke_url)
        if payload.get("stream", True) is False:
            payload = {**payload, "stream": True}
        self.last_inputs = {
            "url": invoke_url,
            "headers": self.headers["stream"],
            "json": self.payload_fn(payload),
        }
        async with self.get_asession_fn() as session:
            async with session.post(**self.last_inputs) as response:
                self._try_raise(response)
                async for line in response.content.iter_any():
                    if line and line.strip() != b"data: [DONE]":
                        line = line.decode("utf-8")
                        msg, final_line = self.postprocess(line, stop=stop)
                        yield msg
                        if final_line:
                            break


class _NVIDIAClient(BaseModel):
    """
    Higher-Level AI Foundation Model Function API Client with argument defaults.
    Is subclassed by ChatNVIDIA to provide a simple LangChain interface.
    """

    client: NVEModel = Field(NVEModel)

    model: str = Field(..., description="Name of the model to invoke")
    is_hosted: bool = Field(True)

    ####################################################################################

    @root_validator(pre=True)
    def _preprocess_args(cls, values: Any) -> Any:
        values["client"] = NVEModel(**values)

        if "base_url" in values:
            values["is_hosted"] = urlparse(values["base_url"]).netloc in [
                "integrate.api.nvidia.com",
                "ai.api.nvidia.com",
            ]

        return values

    @root_validator
    def _postprocess_args(cls, values: Any) -> Any:
        if values["is_hosted"]:
            if not values["client"].api_key:
                warnings.warn(
                    "An API key is required for the hosted NIM. "
                    "This will become an error in the future.",
                    UserWarning,
                )

            name = values.get("model")
            if model := determine_model(name):
                values["model"] = model.id
                # not all models are on https://integrate.api.nvidia.com/v1,
                # those that are not are served from their own endpoints
                if model.endpoint:
                    # we override the infer_path to use the custom endpoint
                    values["client"].infer_path = model.endpoint
            else:
                if not (client := values.get("client")):
                    warnings.warn(f"Unable to determine validity of {name}")
                else:
                    if any(model.id == name for model in client.available_models):
                        warnings.warn(
                            f"Found {name} in available_models, but type is "
                            "unknown and inference may fail."
                        )
                    else:
                        raise ValueError(
                            f"Model {name} is unknown, check `available_models`"
                        )

        return values

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"api_key": self.client._api_key_var}

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}
        if getattr(self.client, "base_url"):
            attributes["base_url"] = self.client.base_url

        if self.model:
            attributes["model"] = self.model

        return attributes

    def get_available_models(
        self,
        filter: str,
        **kwargs: Any,
    ) -> List[Model]:
        """Retrieve a list of available models."""
        available = [
            model for model in self.client.available_models if model.client == filter
        ]

        # if we're talking to a hosted endpoint, we mix in the known models
        # because they are not all discoverable by listing. for instance,
        # the NV-Embed-QA and VLM models are hosted on ai.api.nvidia.com
        # instead of integrate.api.nvidia.com.
        if self.is_hosted:
            known = set(
                model for model in MODEL_TABLE.values() if model.client == filter
            )
            available = list(set(available) | known)

        return available
