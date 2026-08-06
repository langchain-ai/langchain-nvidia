# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile-only integration-test marker used by repository CI."""

import pytest


@pytest.mark.compile
def test_placeholder() -> None:
    """Allow CI to collect the integration-test package without paid calls."""
