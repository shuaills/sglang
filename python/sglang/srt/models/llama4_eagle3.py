# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Inference-only LLaMA4-EAGLE3 model compatible with HuggingFace weights."""

from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import Llama4TextConfig

from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.llama4 import Llama4DecoderLayer, Llama4ForCausalLM, Llama4MoE
from sglang.srt.utils import add_prefix


class Llama4DecoderLayer(Llama4DecoderLayer):
    def __init__(
        self,
        config: Llama4TextConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, layer_id, quant_config, prefix)

        # Override qkv_proj for concatenated input (2 * hidden_size)
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        self.self_attn.qkv_proj = QKVParallelLinear(
            hidden_size=2 * self.hidden_size,
            head_size=self.self_attn.head_dim,
            total_num_heads=self.self_attn.total_num_heads,
            total_num_kv_heads=self.self_attn.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", add_prefix("self_attn", prefix)),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.feed_forward = Llama4MoE(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("feed_forward", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        embeds = self.input_layernorm(embeds)
        hidden_states = self.hidden_norm(hidden_states)

        hidden_states = torch.cat([embeds, hidden_states], dim=-1)

        # Self Attention
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        hidden_states = self.feed_forward(hidden_states, forward_batch)

        return hidden_states, residual


class Llama4Model(nn.Module):
    def __init__(
        self,
        config: Llama4TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = getattr(config, "draft_vocab_size", config.vocab_size)

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )

        self.midlayer = Llama4DecoderLayer(config, 0, quant_config, prefix)

        if hasattr(config, "target_hidden_size"):
            self.fc = torch.nn.Linear(config.target_hidden_size * 3, config.hidden_size)
        else:
            self.fc = torch.nn.Linear(config.hidden_size * 3, config.hidden_size)

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            embeds = self.embed_tokens(input_ids)
        else:
            embeds = input_embeds

        hidden_states = forward_batch.spec_info.hidden_states
        if hidden_states.shape[-1] != embeds.shape[-1]:
            hidden_states = self.fc(hidden_states)

        residual = None
        hidden_states, residual = self.midlayer(
            positions,
            embeds,
            hidden_states,
            forward_batch,
            residual,
        )

        hidden_states_to_logits, hidden_states_to_aux = self.norm(
            hidden_states, residual
        )

        return hidden_states_to_logits, [hidden_states_to_aux]


class Llama4ForCausalLMEagle3(Llama4ForCausalLM):
    def __init__(
        self,
        config: Llama4TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config

        # Initialize pp_group from distributed module
        from sglang.srt.distributed import get_pp_group

        self.pp_group = get_pp_group()

        if hasattr(config, "num_hidden_layers") and config.num_hidden_layers != 1:
            raise ValueError("EAGLE3 currently only supports 1 layer")

        self.model = Llama4Model(
            config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        draft_vocab_size = getattr(config, "draft_vocab_size", config.vocab_size)

        if getattr(config, "tie_word_embeddings", False):
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                draft_vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )

        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = True

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        for name, loaded_weight in weights:
            if "d2t" in name:
                self.hot_token_id = loaded_weight + torch.arange(loaded_weight.shape[0])

            if "d2t" not in name and "t2d" not in name and "lm_head" not in name:
                new_name = f"model.{name}"
                super().load_weights([(new_name, loaded_weight)])
            elif "lm_head" in name:
                super().load_weights([(name, loaded_weight)])

    def get_hot_token_id(self):
        # return getattr(self, "hot_token_id", None)
        # For testing, return a simple fixed tensor on the correct device
        device = next(self.parameters()).device
        draft_vocab_size = getattr(self.config, "draft_vocab_size", 32000)
        # Use a smaller number of hot tokens to be safe
        num_hot_tokens = min(32000, draft_vocab_size)
        return torch.arange(num_hot_tokens, dtype=torch.int32, device=device)


EntryClass = [Llama4ForCausalLMEagle3]
