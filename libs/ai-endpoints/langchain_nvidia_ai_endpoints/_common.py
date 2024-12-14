from __future__ import annotations

import json
import logging
import os
import time
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from urllib.parse import urlparse, urlunparse

import requests
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    field_validator,
)
from requests.models import Response

from langchain_nvidia_ai_endpoints._statics import MODEL_TABLE, Model, determine_model

logger = logging.getLogger(__name__)

_API_KEY_VAR = "NVIDIA_API_KEY"
_BASE_URL_VAR = "NVIDIA_BASE_URL"


class _NVIDIAClient(BaseModel):
    """
    Low level client library interface to NIM endpoints.
    """

    default_hosted_model_name: str = Field(..., description="Default model name to use")
    # "mdl_name" because "model_" is a protected namespace in pydantic
    mdl_name: Optional[str] = Field(..., description="Name of the model to invoke")
    model: Optional[Model] = Field(None, description="The model to invoke")
    is_hosted: bool = Field(True)
    cls: str = Field(..., description="Class Name")

    # todo: add a validator for requests.Response (last_response attribute) and
    #       remove arbitrary_types_allowed=True
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    ## Core defaults. These probably should not be changed
    base_url: str = Field(
        default_factory=lambda: os.getenv(
            _BASE_URL_VAR, "https://integrate.api.nvidia.com/v1"
        ),
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
    polling_url_tmpl: str = Field(
        "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/{request_id}",
        description="Path for polling after HTTP 202 responses",
    )
    get_session_fn: Callable = Field(requests.Session)

    api_key: Optional[SecretStr] = Field(
        default_factory=lambda: SecretStr(
            os.getenv(_API_KEY_VAR, "INTERNAL_LCNVAIE_ERROR")
        )
        if _API_KEY_VAR in os.environ
        else None,
        description="API Key for service of choice",
    )

    ## Generation arguments
    timeout: float = Field(
        60,
        ge=0,
        description="The minimum amount of time (in sec) to poll after a 202 response",
    )
    interval: float = Field(
        0.02,
        ge=0,
        description="Interval (in sec) between polling attempts after a 202 response",
    )
    last_inputs: Optional[dict] = Field(
        default={}, description="Last inputs sent over to the server"
    )
    last_response: Optional[Response] = Field(
        None, description="Last response sent from the server"
    )
    headers_tmpl: dict = Field(
        {
            "call": {
                "Accept": "application/json",
                "Authorization": "Bearer **********",
                "User-Agent": "langchain-nvidia-ai-endpoints",
            },
            "stream": {
                "Accept": "text/event-stream",
                "Content-Type": "application/json",
                "Authorization": "Bearer **********",
                "User-Agent": "langchain-nvidia-ai-endpoints",
            },
        },
        description="Headers template must contain `call` and `stream` keys.",
    )
    _available_models: Optional[List[Model]] = PrivateAttr(default=None)

    ###################################################################################
    ################### Validation and Initialization #################################

    @field_validator("base_url")
    def _validate_base_url(cls, v: str) -> str:
        """
        validate the base_url.

        if the base_url is not a url, raise an error

        if the base_url does not end in /v1, e.g. /embeddings, /completions, /rankings,
        or /reranking, emit a warning. old documentation told users to pass in the full
        inference url, which is incorrect and prevents model listing from working.

        normalize base_url to end in /v1
        """
        ## Making sure /v1 in added to the url
        if v is not None:
            parsed = urlparse(v)

            # Ensure scheme and netloc (domain name) are present
            if not (parsed.scheme and parsed.netloc):
                expected_format = "Expected format is: http://host:port"
                raise ValueError(f"Invalid base_url format. {expected_format} Got: {v}")

            normalized_path = parsed.path.rstrip("/")
            if not normalized_path.endswith("/v1"):
                warnings.warn(
                    f"{v} does not end in /v1, you may "
                    "have inference and listing issues"
                )
                normalized_path += "/v1"

            v = urlunparse(
                (parsed.scheme, parsed.netloc, normalized_path, None, None, None)
            )

        return v

    # final validation after model is constructed
    # todo: when pydantic v2 is available,
    #       use __post_init__ or model_validator(method="after")
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.is_hosted = urlparse(self.base_url).netloc in [
            "integrate.api.nvidia.com",
            "ai.api.nvidia.com",
        ]

        if self.is_hosted:
            if not self.api_key:
                warnings.warn(
                    "An API key is required for the hosted NIM. "
                    "This will become an error in the future.",
                    UserWarning,
                )

            # set default model for hosted endpoint
            if not self.mdl_name:
                self.mdl_name = self.default_hosted_model_name

            if model := determine_model(self.mdl_name):
                if not model.client:
                    warnings.warn(f"Unable to determine validity of {model.id}")
                elif model.client != self.cls:
                    raise ValueError(
                        f"Model {model.id} is incompatible with client {self.cls}. "
                        f"Please check `{self.cls}.get_available_models()`."
                    )

                # not all models are on https://integrate.api.nvidia.com/v1,
                # those that are not are served from their own endpoints
                if model.endpoint:
                    # we override the infer_path to use the custom endpoint
                    self.infer_path = model.endpoint
            else:
                candidates = [
                    model
                    for model in self.available_models
                    if model.id == self.mdl_name
                ]
                assert len(candidates) <= 1, (
                    f"Multiple candidates for {self.mdl_name} "
                    f"in `available_models`: {candidates}"
                )
                if candidates:
                    model = candidates[0]
                    warnings.warn(
                        f"Found {self.mdl_name} in available_models, but type is "
                        "unknown and inference may fail."
                    )
                else:
                    if self.mdl_name.startswith("nvdev/"):  # assume valid
                        model = Model(id=self.mdl_name)
                    else:
                        raise ValueError(
                            f"Model {self.mdl_name} is unknown, "
                            "check `available_models`"
                        )
            self.model = model
            self.mdl_name = self.model.id  # name may change because of aliasing
        else:
            # set default model
            if not self.mdl_name:
                valid_models = [
                    model
                    for model in self.available_models
                    if not model.base_model or model.base_model == model.id
                ]
                self.model = next(iter(valid_models), None)
                if self.model:
                    self.mdl_name = self.model.id
                    warnings.warn(
                        f"Default model is set as: {self.mdl_name}. \n"
                        "Set model using model parameter. \n"
                        "To get available models use available_models property.",
                        UserWarning,
                    )
                else:
                    raise ValueError("No locally hosted model was found.")

    ###################################################################################
    ################### LangChain functions ###########################################

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"api_key": _API_KEY_VAR}

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}
        attributes["base_url"] = self.base_url

        if self.mdl_name:
            attributes["model"] = self.mdl_name

        return attributes

    ###################################################################################
    ################### Property accessors ############################################

    @property
    def infer_url(self) -> str:
        return self.infer_path.format(base_url=self.base_url)

    ###################################################################################
    ################### Authorization handling ########################################

    def __add_authorization(self, payload: dict) -> dict:
        if self.api_key:
            payload = {**payload}
            payload["headers"] = {
                **payload.get("headers", {}),
                "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            }
        return payload

    ###################################################################################
    ################### Model discovery and selection #################################

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

            # add base model for local-nim mode
            model.base_model = element.get("root")

            self._available_models.append(model)

        return self._available_models

    def get_available_models(
        self,
        filter: str,
        **kwargs: Any,
    ) -> List[Model]:
        """Retrieve a list of available models."""

        available = self.available_models

        # if we're talking to a hosted endpoint, we mix in the known models
        # because they are not all discoverable by listing. for instance,
        # the NV-Embed-QA and VLM models are hosted on ai.api.nvidia.com
        # instead of integrate.api.nvidia.com.
        if self.is_hosted:
            known = set(MODEL_TABLE.values())
            available = [
                model for model in set(available) | known if model.client == filter
            ]

        return available

    ###################################################################################
    ## Core utilities for posting and getting from NV Endpoints #######################

    def _post(
        self,
        invoke_url: str,
        payload: Optional[dict] = {},
        extra_headers: dict = {},
    ) -> Tuple[Response, requests.Session]:
        """Method for posting to the AI Foundation Model Function API."""
        self.last_inputs = {
            "url": invoke_url,
            "headers": {
                **self.headers_tmpl["call"],
                **extra_headers,
            },
            "json": payload,
        }
        session = self.get_session_fn()
        self.last_response = response = session.post(
            **self.__add_authorization(self.last_inputs)
        )
        self._try_raise(response)
        return response, session

    def _get(
        self,
        invoke_url: str,
    ) -> Tuple[Response, requests.Session]:
        """Method for getting from the AI Foundation Model Function API."""
        self.last_inputs = {
            "url": invoke_url,
            "headers": self.headers_tmpl["call"],
        }
        session = self.get_session_fn()
        self.last_response = response = session.get(
            **self.__add_authorization(self.last_inputs)
        )
        self._try_raise(response)
        return response, session

    def _wait(self, response: Response, session: requests.Session) -> Response:
        """
        Any request may return a 202 status code, which means the request is still
        processing. This method will wait for a response using the request id.

        see https://docs.nvidia.com/cloud-functions/user-guide/latest/cloud-function/api.html#http-polling
        """
        start_time = time.time()
        # note: the local NIM does not return a 202 status code
        #       (per RL 22may2024 circa 24.05)
        while response.status_code == 202:
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
            payload = {
                "url": self.polling_url_tmpl.format(request_id=request_id),
                "headers": self.headers_tmpl["call"],
            }
            self.last_response = response = session.get(
                **self.__add_authorization(payload)
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

    ###################################################################################
    ## Generation interface to allow users to generate new values from endpoints ######

    def get_req(
        self,
        payload: dict = {},
        extra_headers: dict = {},
    ) -> Response:
        """Post to the API."""
        response, session = self._post(
            self.infer_url, payload, extra_headers=extra_headers
        )
        return self._wait(response, session)

    def postprocess(
        self,
        response: Union[str, Response],
    ) -> Tuple[dict, bool]:
        """Parses a response from the AI Foundation Model Function API.
        Strongly assumes that the API will return a single response.
        """
        return self._aggregate_msgs(self._process_response(response))

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

    def _aggregate_msgs(self, msg_list: Sequence[dict]) -> Tuple[dict, bool]:
        """Dig out relevant details of aggregated message"""
        content_buffer: Dict[str, Any] = dict()
        content_holder: Dict[Any, Any] = dict()
        usage_holder: Dict[Any, Any] = dict()  ####
        finish_reason_holder: Optional[str] = None
        is_stopped = False
        for msg in msg_list:
            usage_holder = msg.get("usage", {})  ####
            if "choices" in msg:
                ## Tease out ['choices'][0]...['delta'/'message']
                # when streaming w/ usage info, we may get a response
                #  w/ choices: [] that includes final usage info
                choices = msg.get("choices", [{}])
                msg = choices[0] if choices else {}
                # todo: this meeds to be fixed, the fact we only
                #       use the first choice breaks the interface
                finish_reason_holder = msg.get("finish_reason", None)
                is_stopped = finish_reason_holder == "stop"
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
        if finish_reason_holder:
            content_holder.update(finish_reason=finish_reason_holder)
        return content_holder, is_stopped

    ###################################################################################
    ## Streaming interface to allow you to iterate through progressive generations ####

    def get_req_stream(
        self,
        payload: dict,
        extra_headers: dict = {},
    ) -> Iterator[Dict]:
        self.last_inputs = {
            "url": self.infer_url,
            "headers": {
                **self.headers_tmpl["stream"],
                **extra_headers,
            },
            "json": payload,
        }

        response = self.get_session_fn().post(
            stream=True, **self.__add_authorization(self.last_inputs)
        )
        self._try_raise(response)
        call: _NVIDIAClient = self.model_copy()

        def out_gen() -> Generator[dict, Any, Any]:
            ## Good for client, since it allows self.last_inputs
            for line in response.iter_lines():
                if line and line.strip() != b"data: [DONE]":
                    line = line.decode("utf-8")
                    msg, final_line = call.postprocess(line)
                    yield msg
                    if final_line:
                        break
                self._try_raise(response)

        return (r for r in out_gen())
