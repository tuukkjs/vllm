# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the DSA indexer's expanded-block-table sizing.

The expanded block-table workspace must be wide enough for the runner's aligned
block table, including speculative lookahead. It is also used in CUDA graph
paths, so it must be allocated at the final width up front rather than replaced
later during decode.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadataBuilder,
)

_expanded_block_table = DeepseekV32IndexerMetadataBuilder._expanded_block_table
_max_expanded_block_table_width = (
    DeepseekV32IndexerMetadataBuilder._max_expanded_block_table_width
)


class _FakeKVCacheSpec:
    def __init__(self, block_size: int):
        self.block_size = block_size

    def max_num_blocks_per_req(self, vllm_config, max_len: int) -> int:
        del vllm_config
        return max_len // self.block_size


def _fake_builder(rows: int, width: int) -> SimpleNamespace:
    return SimpleNamespace(
        expanded_block_table_buffer=torch.zeros((rows, width), dtype=torch.int32),
        device=torch.device("cpu"),
    )


def test_max_expanded_block_table_width_aligns_speculative_tokens():
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=20000),
        scheduler_config=SimpleNamespace(max_num_encoder_input_tokens=0),
    )

    width = _max_expanded_block_table_width(
        vllm_config,
        _FakeKVCacheSpec(block_size=1),
        num_speculative_tokens=2,
    )

    assert width == 20096


def test_expanded_block_table_reuses_preallocated_width():
    # The runner may hand in a narrower table than the conservative workspace.
    # The workspace address must still stay stable for CUDA graph replay.
    fake = _fake_builder(rows=4, width=16384)
    original = fake.expanded_block_table_buffer

    first = _expanded_block_table(fake, 8192)
    second = _expanded_block_table(fake, 8192)

    assert first is original
    assert second is original
    assert first.shape == (4, 16384)


def test_expanded_block_table_stable_when_width_matches():
    fake = _fake_builder(rows=4, width=8192)
    original = fake.expanded_block_table_buffer

    first = _expanded_block_table(fake, 8192)
    second = _expanded_block_table(fake, 8192)

    assert first is original
    assert second is original


def test_expanded_block_table_rejects_wider_runner_width():
    fake = _fake_builder(rows=4, width=8192)
    original = fake.expanded_block_table_buffer

    with pytest.raises(RuntimeError, match="too narrow"):
        _expanded_block_table(fake, 16384)

    assert fake.expanded_block_table_buffer is original
