# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route LangChain model calls through Switchyard's libsy algorithms."""

from .client import LangChainLlmClient as LangChainLlmClient
from .middleware import SwitchyardRoutingMiddleware as SwitchyardRoutingMiddleware

__all__ = [
    "LangChainLlmClient",
    "SwitchyardRoutingMiddleware",
]
