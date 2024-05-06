from __future__ import annotations

import gc
import json
import queue
import random
import time
from functools import partial
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import google.protobuf.json_format
import numpy as np
import tensorrt_llm
import torch
import tritonclient.grpc as grpcclient
from langchain_core.callbacks import CallbackManager, CallbackManagerForLLMRun
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import Field, PrivateAttr, root_validator
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import ModelRunner, ModelRunnerCpp
from tritonclient.grpc.service_pb2 import ModelInferResponse
from tritonclient.utils import np_to_triton_dtype

from .utils import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
    load_tokenizer,
    read_model_name,
)


class TritonTensorRTError(Exception):
    """Base exception for TritonTensorRT."""


class TritonTensorRTRuntimeError(TritonTensorRTError, RuntimeError):
    """Runtime error for TritonTensorRT."""


class TritonTensorRTLLM(BaseLLM):
    """TRTLLM triton models.

    Arguments:
        server_url: (str) The URL of the Triton inference server to use.
        model_name: (str) The name of the Triton TRT model to use.
        temperature: (str) Temperature to use for sampling
        top_p: (float) The top-p value to use for sampling
        top_k: (float) The top k values use for sampling
        beam_width: (int) Last n number of tokens to penalize
        repetition_penalty: (int) Last n number of tokens to penalize
        length_penalty: (float) The penalty to apply repeated tokens
        tokens: (int) The maximum number of tokens to generate.
        client: The client object used to communicate with the inference server

    Example:
        .. code-block:: python

            from langchain_nvidia_trt import TritonTensorRTLLM

            model = TritonTensorRTLLM()


    """

    server_url: Optional[str] = Field(None, alias="server_url")
    model_name: str = Field(
        ..., description="The name of the model to use, such as 'ensemble'."
    )
    ## Optional args for the model
    temperature: float = 1.0
    top_p: float = 0
    top_k: int = 1
    tokens: int = 100
    beam_width: int = 1
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    client: grpcclient.InferenceServerClient
    stop: List[str] = Field(
        default_factory=lambda: ["</s>"], description="Stop tokens."
    )
    seed: int = Field(42, description="The seed to use for random generation.")
    load_model: bool = Field(
        True,
        description="Request the inference server to load the specified model.\
            Certain Triton configurations do not allow for this operation.",
    )

    def __del__(self):
        """Ensure the client streaming connection is properly shutdown"""
        self.client.close()

    @root_validator(pre=True, allow_reuse=True)
    def validate_environment(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that python package exists in environment."""
        if not values.get("client"):
            values["client"] = grpcclient.InferenceServerClient(values["server_url"])
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "nvidia-trt-llm"

    @property
    def _model_default_parameters(self) -> Dict[str, Any]:
        return {
            "tokens": self.tokens,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
            "length_penalty": self.length_penalty,
            "beam_width": self.beam_width,
        }

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get all the identifying parameters."""
        return {
            "server_url": self.server_url,
            "model_name": self.model_name,
            **self._model_default_parameters,
        }

    def _get_invocation_params(self, **kwargs: Any) -> Dict[str, Any]:
        return {**self._model_default_parameters, **kwargs}

    def get_model_list(self) -> List[str]:
        """Get a list of models loaded in the triton server."""
        res = self.client.get_model_repository_index(as_json=True)
        return [model["name"] for model in res["models"]]

    def _load_model(self, model_name: str, timeout: int = 1000) -> None:
        """Load a model into the server."""
        if self.client.is_model_ready(model_name):
            return

        self.client.load_model(model_name)
        t0 = time.perf_counter()
        t1 = t0
        while not self.client.is_model_ready(model_name) and t1 - t0 < timeout:
            t1 = time.perf_counter()

        if not self.client.is_model_ready(model_name):
            raise TritonTensorRTRuntimeError(
                f"Failed to load {model_name} on Triton in {timeout}s"
            )

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        self._load_model(self.model_name)

        invocation_params = self._get_invocation_params(**kwargs)
        stop_words = stop if stop is not None else self.stop
        generations = []
        # TODO: We should handle the native batching instead.
        for prompt in prompts:
            invoc_params = {**invocation_params, "prompt": [[prompt]]}
            result: str = self._request(
                self.model_name,
                stop=stop_words,
                **invoc_params,
            )
            generations.append([Generation(text=result, generation_info={})])
        return LLMResult(generations=generations)

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        self._load_model(self.model_name)

        invocation_params = self._get_invocation_params(**kwargs, prompt=[[prompt]])
        stop_words = stop if stop is not None else self.stop

        inputs = self._generate_inputs(stream=True, **invocation_params)
        outputs = self._generate_outputs()

        result_queue = self._invoke_triton(self.model_name, inputs, outputs, stop_words)

        for token in result_queue:
            yield GenerationChunk(text=token)
            if run_manager:
                run_manager.on_llm_new_token(token)

        self.client.stop_stream()

    ##### BELOW ARE METHODS PREVIOUSLY ONLY IN THE GRPC CLIENT

    def _request(
        self,
        model_name: str,
        prompt: Sequence[Sequence[str]],
        stop: Optional[List[str]] = None,
        **params: Any,
    ) -> str:
        """Request inferencing from the triton server."""
        # create model inputs and outputs
        inputs = self._generate_inputs(stream=False, prompt=prompt, **params)
        outputs = self._generate_outputs()

        result_queue = self._invoke_triton(self.model_name, inputs, outputs, stop)

        result_str = ""
        try:
            for token in result_queue:
                if isinstance(token, Exception):
                    raise token
                result_str += token
        finally:
            self.client.stop_stream()

        return result_str

    def _invoke_triton(self, model_name, inputs, outputs, stop_words):
        if not self.client.is_model_ready(model_name):
            raise RuntimeError("Cannot request streaming, model is not loaded")

        request_id = str(random.randint(1, 9999999))  # nosec

        result_queue = StreamingResponseGenerator(
            self,
            request_id,
            force_batch=False,
            stop_words=stop_words,
        )

        self.client.start_stream(
            callback=partial(
                self._stream_callback,
                result_queue,
                stop_words=stop_words,
            )
        )

        # Even though this request may not be a streaming request certain configurations
        # in Triton prevent the GRPC server from accepting none streaming connections.
        # Therefore we call the streaming API and combine the streamed results.
        self.client.async_stream_infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
            request_id=request_id,
        )

        return result_queue

    def _generate_outputs(
        self,
    ) -> List[grpcclient.InferRequestedOutput]:
        """Generate the expected output structure."""
        return [grpcclient.InferRequestedOutput("text_output")]

    def _prepare_tensor(
        self, name: str, input_data: np.ndarray
    ) -> grpcclient.InferInput:
        """Prepare an input data structure."""

        t = grpcclient.InferInput(
            name, input_data.shape, np_to_triton_dtype(input_data.dtype)
        )
        t.set_data_from_numpy(input_data)
        return t

    def _generate_inputs(
        self,
        prompt: Sequence[Sequence[str]],
        tokens: int = 300,
        temperature: float = 1.0,
        top_k: float = 1,
        top_p: float = 0,
        beam_width: int = 1,
        repetition_penalty: float = 1,
        length_penalty: float = 1.0,
        stream: bool = True,
    ) -> List[grpcclient.InferRequestedOutput]:
        """Create the input for the triton inference server."""
        query = np.array(prompt).astype(object)
        request_output_len = np.array([tokens]).astype(np.uint32).reshape((1, -1))
        runtime_top_k = np.array([top_k]).astype(np.uint32).reshape((1, -1))
        runtime_top_p = np.array([top_p]).astype(np.float32).reshape((1, -1))
        temperature_array = np.array([temperature]).astype(np.float32).reshape((1, -1))
        len_penalty = np.array([length_penalty]).astype(np.float32).reshape((1, -1))
        repetition_penalty_array = (
            np.array([repetition_penalty]).astype(np.float32).reshape((1, -1))
        )
        random_seed = np.array([self.seed]).astype(np.uint64).reshape((1, -1))
        beam_width_array = np.array([beam_width]).astype(np.uint32).reshape((1, -1))
        streaming_data = np.array([[stream]], dtype=bool)

        inputs = [
            self._prepare_tensor("text_input", query),
            self._prepare_tensor("max_tokens", request_output_len),
            self._prepare_tensor("top_k", runtime_top_k),
            self._prepare_tensor("top_p", runtime_top_p),
            self._prepare_tensor("temperature", temperature_array),
            self._prepare_tensor("length_penalty", len_penalty),
            self._prepare_tensor("repetition_penalty", repetition_penalty_array),
            self._prepare_tensor("random_seed", random_seed),
            self._prepare_tensor("beam_width", beam_width_array),
            self._prepare_tensor("stream", streaming_data),
        ]
        return inputs

    def _send_stop_signals(self, model_name: str, request_id: str) -> None:
        """Send the stop signal to the Triton Inference server."""
        stop_inputs = self._generate_stop_signals()
        self.client.async_stream_infer(
            model_name,
            stop_inputs,
            request_id=request_id,
            parameters={"Streaming": True},
        )

    def _generate_stop_signals(
        self,
    ) -> List[grpcclient.InferInput]:
        """Generate the signal to stop the stream."""
        inputs = [
            grpcclient.InferInput("input_ids", [1, 1], "INT32"),
            grpcclient.InferInput("input_lengths", [1, 1], "INT32"),
            grpcclient.InferInput("request_output_len", [1, 1], "UINT32"),
            grpcclient.InferInput("stop", [1, 1], "BOOL"),
        ]
        inputs[0].set_data_from_numpy(np.empty([1, 1], dtype=np.int32))
        inputs[1].set_data_from_numpy(np.zeros([1, 1], dtype=np.int32))
        inputs[2].set_data_from_numpy(np.array([[0]], dtype=np.uint32))
        inputs[3].set_data_from_numpy(np.array([[True]], dtype="bool"))
        return inputs

    @staticmethod
    def _process_result(result: Dict[str, str]) -> str:
        """Post-process the result from the server."""

        message = ModelInferResponse()
        google.protobuf.json_format.Parse(json.dumps(result), message)
        infer_result = grpcclient.InferResult(message)
        np_res = infer_result.as_numpy("text_output")

        generated_text = ""
        if np_res is not None:
            generated_text = "".join([token.decode() for token in np_res])

        return generated_text

    def _stream_callback(
        self,
        result_queue: queue.Queue[Union[Optional[Dict[str, str]], str]],
        result: grpcclient.InferResult,
        error: str,
        stop_words: List[str],
    ) -> None:
        """Add streamed result to queue."""
        if error:
            result_queue.put(error)
        else:
            response_raw: dict = result.get_response(as_json=True)
            # TODO: Check the response is a map rather than a string
            if "outputs" in response_raw:
                # the very last response might have no output, just the final flag
                response = self._process_result(response_raw)

                if response in stop_words:
                    result_queue.put(None)
                else:
                    result_queue.put(response)

            if response_raw["parameters"]["triton_final_response"]["bool_param"]:
                # end of the generation
                result_queue.put(None)

    def stop_stream(
        self, model_name: str, request_id: str, signal: bool = True
    ) -> None:
        """Close the streaming connection."""
        if signal:
            self._send_stop_signals(model_name, request_id)
        self.client.stop_stream()


class StreamingResponseGenerator(queue.Queue):
    """A Generator that provides the inference results from an LLM."""

    def __init__(
        self,
        llm: TritonTensorRTLLM,
        request_id: str,
        force_batch: bool,
        stop_words: Sequence[str],
    ) -> None:
        """Instantiate the generator class."""
        super().__init__()
        self.llm = llm
        self.request_id = request_id
        self._batch = force_batch
        self._stop_words = stop_words

    def __iter__(self) -> StreamingResponseGenerator:
        """Return self as a generator."""
        return self

    def __next__(self) -> str:
        """Return the next retrieved token."""
        val = self.get()
        if val is None or val in self._stop_words:
            self.llm.stop_stream(
                self.llm.model_name, self.request_id, signal=not self._batch
            )
            raise StopIteration()
        return val

class TrtLlmAPI(BaseLLM):
    model_path: Optional[str] = Field(
        description="The path to the trt engine."
    )
    tokenizer_dir: Optional[str] = Field(
        description="The path to the trt engine."
    )
    temperature: float = Field(
        default=0.1, description="The temperature to use for sampling."
    )
    max_new_tokens: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="The maximum number of tokens to generate."
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The maximum number of context tokens for the model."
    )
    verbose: bool = Field(
        default=False,
        description="Whether to print verbose output."
    )

    _model: Any = PrivateAttr()
    _model_name = PrivateAttr()
    _model_version = PrivateAttr()
    _model_config: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _pad_id:Any = PrivateAttr()
    _end_id: Any = PrivateAttr()
    _max_new_tokens = PrivateAttr()
    _max_input_tokens = PrivateAttr()
    _sampling_config = PrivateAttr()
    _debug_mode = PrivateAttr()
    _add_special_tokens = PrivateAttr()
    _verbose = PrivateAttr()

    def _init_attr(
            self,
            model_path: Optional[str] = None,
            tokenizer_dir: Optional[str] = None,
            vocab_file: Optional[str] = None,
            temperature: float = 0.1,
            max_new_tokens: int = DEFAULT_NUM_OUTPUTS,
            context_window: int = DEFAULT_CONTEXT_WINDOW,
            callback_manager: Optional[CallbackManager] = None,
            use_py_session = True,
            add_special_tokens = False,
            trtLlm_debug_mode = False,
            verbose: bool = False
    ) -> None:
        runtime_rank = tensorrt_llm.mpi_rank()
        self._model_name, self._model_version = read_model_name(model_path)
        if tokenizer_dir is None:
            logger.error(
                "tokenizer_dir is not specified."
            )

        self._max_input_tokens=context_window
        self._add_special_tokens=add_special_tokens
        self._verbose = verbose

        self._tokenizer, self._pad_id, self._end_id = load_tokenizer(
            tokenizer_dir=tokenizer_dir,
            vocab_file=vocab_file,
            model_name=self._model_name,
            model_version=self._model_version,
        )

        runner_cls = ModelRunner if use_py_session else ModelRunnerCpp
        if verbose:
            logger.info(f"Trt-llm mode debug mode: {trtLlm_debug_mode}")

        runner_kwargs = dict(engine_dir=model_path,
                             rank=runtime_rank,
                             debug_mode=trtLlm_debug_mode,
                             lora_ckpt_source='hf')

        if not use_py_session:
            runner_kwargs.update(free_gpu_memory_fraction = 0.5)

        self._model = runner_cls.from_dir(**runner_kwargs)

        self._max_new_tokens = max_new_tokens

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "nvidia-trt-llm-api"

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        generations = []
        for prompt in prompts:
            text = (
                self.complete_call(prompt, stop=stop, run_manager=run_manager, **kwargs)
            )
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    def complete_call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        self._init_attr(model_path=self.model_path,
                        tokenizer_dir=self.tokenizer_dir,
                        verbose=self.verbose,
                        temperature=self.temperature)

        if self._verbose:
            logger.info(f"Context send to LLM \n: {prompt}")

        input_text = [prompt]
        batch_input_ids = self.parse_input(
                                tokenizer=self._tokenizer,
                                input_text=input_text,
                                prompt_template=None,
                                input_file=None,
                                add_special_tokens=self._add_special_tokens,
                                max_input_length=self._max_input_tokens,
                                pad_id=self._pad_id,
                                num_prepend_vtokens=None,
                                model_name= self._model_name,
                                model_version=self._model_version)
        input_lengths = [x.size(0) for x in batch_input_ids]

        if self._verbose:
            logger.info(f"Number of token : {input_lengths[0]}")

        with torch.no_grad():
            outputs = self._model.generate(
                batch_input_ids,
                max_new_tokens=self._max_new_tokens,
                max_attention_window_size=4096,
                end_id=self._end_id,
                pad_id=self._pad_id,
                temperature=self.temperature,
                top_k=1,
                top_p=0,
                num_beams=1,
                length_penalty=1.0,
                early_stopping=False,
                repetition_penalty=1.0,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                stop_words_list=None,
                bad_words_list=None,
                lora_uids=None,
                prompt_table_path=None,
                prompt_tasks=None,
                streaming=False,
                output_sequence_lengths=True,
                return_dict=True)
            torch.cuda.synchronize()

        output_ids = outputs['output_ids']
        sequence_lengths = outputs['sequence_lengths']
        output_txt, output_token_ids = self.print_output(self._tokenizer,
                                                        output_ids,
                                                        input_lengths,
                                                        sequence_lengths)
        # call garbage collected after inference
        torch.cuda.empty_cache()
        gc.collect()
        return output_txt

    def parse_input(self,
                    tokenizer,
                    input_text=None,
                    prompt_template=None,
                    input_file=None,
                    add_special_tokens=False,
                    max_input_length=4096,
                    pad_id=None,
                    num_prepend_vtokens=[],
                    model_name=None,
                    model_version=None):
        if pad_id is None:
            pad_id = tokenizer.pad_token_id

        batch_input_ids = []
        if input_file is None:
            for curr_text in input_text:
                if prompt_template is not None:
                    curr_text = prompt_template.format(input_text=curr_text)
                input_ids = tokenizer.encode(curr_text,
                                             add_special_tokens=add_special_tokens,
                                             truncation=True,
                                             max_length=max_input_length)
                batch_input_ids.append(input_ids)

        if num_prepend_vtokens:
            assert len(num_prepend_vtokens) == len(batch_input_ids)
            base_vocab_size = tokenizer.vocab_size - len(
                tokenizer.special_tokens_map.get('additional_special_tokens', []))
            for i, length in enumerate(num_prepend_vtokens):
                batch_input_ids[i] = list(
                    range(base_vocab_size,
                          base_vocab_size + length)) + batch_input_ids[i]

        if model_name == 'ChatGLMForCausalLM' and model_version == 'glm':
            for ids in batch_input_ids:
                ids.append(tokenizer.sop_token_id)

        batch_input_ids = [
            torch.tensor(x, dtype=torch.int32) for x in batch_input_ids
        ]

        return batch_input_ids

    def print_output(self,
                     tokenizer,
                     output_ids,
                     input_lengths,
                     sequence_lengths,
                     output_csv=None,
                     output_npy=None,
                     context_logits=None,
                     generation_logits=None,
                     output_logits_npy=None):
        output_text = ""
        batch_size, num_beams, _ = output_ids.size()
        if output_csv is None and output_npy is None:
            for batch_idx in range(batch_size):
                for beam in range(num_beams):
                    output_begin = input_lengths[batch_idx]
                    output_end = sequence_lengths[batch_idx][beam]
                    outputs = output_ids[batch_idx][beam][
                              output_begin:output_end].tolist()
                    output_text = tokenizer.decode(outputs)

        output_ids = output_ids.reshape((-1, output_ids.size(2)))
        return output_text, output_ids
