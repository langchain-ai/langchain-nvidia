import warnings
from typing import Any, Dict, Optional

from langchain_nvidia_ai_endpoints._statics import Model, determine_model


def _process_hosted_model(values: Dict) -> None:
    """
    Process logic for a hosted model. Validates compatibility, sets model ID,
    and adjusts client's infer path if necessary.

    Raises:
        ValueError: If the model is incompatible with the client or is unknown.
    """
    name = values["model"]
    cls_name = values["cls"]
    client = values.get("client")

    if client and hasattr(client, "api_key") and not client.api_key:
        warnings.warn(
            "An API key is required for the hosted NIM. "
            "This will become an error in the future.",
            UserWarning,
        )

    model = determine_model(name)
    if model:
        _validate_hosted_model_compatibility(name, cls_name, model)
        values["model"] = model.id
        if model.endpoint:
            values["client"].infer_path = model.endpoint
    else:
        _handle_unknown_hosted_model(name, client)


def _validate_hosted_model_compatibility(
    name: str, cls_name: Optional[str], model: Model
) -> None:
    """
    Validates compatibility of the hosted model with the client.

    Args:
        name (str): The name of the model.
        cls_name (str): The name of the client class.
        model (Any): The model object.
    Raises:
        ValueError: If the model is incompatible with the client.
    """
    if not model.client:
        warnings.warn(f"Unable to determine validity of {name}")
    elif model.client != cls_name:
        raise ValueError(
            f"Model {name} is incompatible with client {cls_name}. "
            f"Please check `{cls_name}.get_available_models()`."
        )


def _handle_unknown_hosted_model(name: str, client: Any) -> None:
    """
    Handles scenarios where the hosted model is unknown or its type is unclear.
    Raises:
        ValueError: If the model is unknown.
    """
    if not client:
        warnings.warn(f"Unable to determine validity of {name}")
    elif any(model.id == name for model in client.available_models):
        warnings.warn(
            f"Found {name} in available_models, but type is "
            "unknown and inference may fail."
        )
    else:
        raise ValueError(f"Model {name} is unknown, check `available_models`.")


def _process_locally_hosted_model(values: Dict) -> None:
    """
    Process logic for a locally hosted model.
    Validates compatibility and sets default model.

    Raises:
        ValueError: If the model is incompatible with the client or is unknown.
    """
    name = values["model"]
    cls_name = values["cls"]
    client = values.get("client")

    if name and isinstance(name, str):
        model = determine_model(name)
        if model:
            _validate_locally_hosted_model_compatibility(name, cls_name, model, client)
        else:
            _handle_unknown_locally_hosted_model(name, client)
    else:
        _set_default_model(values, client)


def _validate_locally_hosted_model_compatibility(
    model_name: str, cls_name: str, model: Model, client: Any
) -> None:
    """
    Validates compatibility of the locally hosted model with the client.

    Args:
        model_name (str): The name of the model.
        cls_name (str): The name of the client class.
        model (Any): The model object.
        client (Any): The client object.

    Raises:
        ValueError: If the model is incompatible with the client or is unknown.
    """
    if model.client != cls_name:
        raise ValueError(
            f"Model {model_name} is incompatible with client {cls_name}. "
            f"Please check `{cls_name}.get_available_models()`."
        )

    if model_name not in [model.id for model in client.available_models]:
        raise ValueError(
            f"Locally hosted {model_name} model was found, check `available_models`."
        )


def _handle_unknown_locally_hosted_model(model_name: str, client: Any) -> None:
    """
    Handles scenarios where the locally hosted model is unknown.

    Raises:
        ValueError: If the model is unknown.
    """
    if model_name not in [model.id for model in client.available_models]:
        raise ValueError(f"Model {model_name} is unknown, check `available_models`.")


def _set_default_model(values: Dict, client: Any) -> None:
    """
    Sets a default model based on client's available models.

    Raises:
        ValueError: If no locally hosted model was found.
    """
    values["model"] = next(
        iter(
            [
                model.id
                for model in client.available_models
                if not model.base_model or model.base_model == model.id
            ]
        ),
        None,
    )
    if values["model"]:
        warnings.warn(
            f'Default model is set as: {values["model"]}. \n'
            "Set model using model parameter. \n"
            "To get available models use available_models property.",
            UserWarning,
        )
    else:
        raise ValueError("No locally hosted model was found.")
