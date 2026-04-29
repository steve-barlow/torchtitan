# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
from pathlib import Path

import torch

_ATTENTION_PATH = (
    Path(__file__).parents[2] / "torchtitan" / "models" / "attention.py"
)
_SPEC = importlib.util.spec_from_file_location("torchtitan_attention_under_test", _ATTENTION_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_ATTENTION_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ATTENTION_MODULE)
ScaledDotProductAttentionWrapper = _ATTENTION_MODULE.ScaledDotProductAttentionWrapper


def test_sdpa_wrapper_accepts_bool_attention_mask() -> None:
    torch.manual_seed(0)
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))[None, None, :, :]

    attention = ScaledDotProductAttentionWrapper()
    output = attention(q, k, v, attn_mask=mask)

    assert output.shape == q.shape
    assert torch.isfinite(output).all()
