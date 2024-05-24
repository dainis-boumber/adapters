from typing import Dict, List, NamedTuple, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from transformers import PretrainedConfig
from transformers.modeling_utils import ModuleUtilsMixin

from ..composition import AdapterCompositionBlock, BatchSplit, Parallel, Stack, adjust_tensors_for_parallel
from ..configuration import ModelAdaptersConfig, PrefixTuningConfig
from ..context import AdapterSetup, ForwardContext
from .adapter_layer_base import ComposableAdapterLayerBase
from .modeling import Activation_Function_Class




class PrefixTuning(nn.Module, ModuleUtilsMixin):
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        input_size: int,
        config: PrefixTuningConfig,
        n_embd_per_head: Optional[int] = None,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.input_size = input_size
        self.n_embd_per_head = n_embd_per_head or self.input_size // self.n_heads
        self.config = config

        self.wte = nn.Embedding(self.config.prefix_length, self.input_size)
        self.control_trans = nn.Sequential(
            nn.Linear(self.input_size, self.config.bottleneck_size),
            Activation_Function_Class(self.config.non_linearity.lower()),
            nn.Linear(self.config.bottleneck_size, self.n_layers * 2 * self.n_heads * self.n_embd_per_head),
        )
        self.dropout = nn.Dropout(self.config.dropout)

    def eject(self):
        input_tokens = torch.arange(self.config.prefix_length).long()
        input_tokens = input_tokens.unsqueeze(0).expand(1, -1).to(self.device)
        embs = self.wte(input_tokens)
        key_values = self.control_trans(embs)  # batch_size x prefix_length x n_layers*2*input_size
        key_values = key_values.view(
            self.config.prefix_length * self.n_layers * 2 * self.input_size
        )  # *2 for key and value

        return key_values

    def forward(self, batch_size):
        input_tokens = torch.arange(self.config.prefix_length).long()
        input_tokens = input_tokens.unsqueeze(0).expand(batch_size, -1).to(self.device)
        embs = self.wte(input_tokens)
        key_values = self.control_trans(embs)  # batch_size x prefix_length x n_layers*2*input_size
        key_values = key_values.view(
            batch_size, self.config.prefix_length, self.n_layers * 2, self.n_heads, self.n_embd_per_head
        )  # *2 for key and value
        key_values = self.dropout(key_values)
        # n_layers * (2 x batch_size x n_heads x prefix_length x n_embd_per_head)
        key_values = key_values.permute(2, 0, 3, 1, 4).split(2)

        return key_values


class FlatPrefixTuning(nn.Module, ModuleUtilsMixin):
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        input_size: int,
        config: PrefixTuningConfig,
        n_embd_per_head: Optional[int] = None,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.input_size = input_size
        self.n_embd_per_head = n_embd_per_head or self.input_size // self.n_heads
        self.config = config

        self.control_trans = nn.Parameter(
            torch.randn(self.config.prefix_length * self.n_layers * 2 * self.n_heads * self.n_embd_per_head)
        )

        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, batch_size):
        key_values = (
            self.control_trans.unsqueeze(0)
            .expand(batch_size, -1)
            .view(batch_size, self.config.prefix_length, self.n_layers * 2, self.n_heads, self.n_embd_per_head)
            .to(self.device)
        )  # *2 for key and value
        key_values = self.dropout(key_values)
        # n_layers * (2 x batch_size x n_heads x prefix_length x n_embd_per_head)
        key_values = key_values.permute(2, 0, 3, 1, 4).split(2)

        return key_values


class PrefixTuningGroup(nn.ModuleDict):
    def __init__(self, module_configs, prefix_tuning_config):
        super().__init__()
        if prefix_tuning_config["flat"]:
            prefix_tuning_class = FlatPrefixTuning
        else:
            prefix_tuning_class = PrefixTuning
        for k, kwargs in module_configs.items():
            self[k] = prefix_tuning_class(**kwargs, config=prefix_tuning_config)

    def eject(self):
        """Converts all PrefixTuning modules into FlatPrefixTuning modules."""
        for k, v in self.items():
            if isinstance(v, PrefixTuning):
                config = v.config.replace(flat=True)
                self[k] = FlatPrefixTuning(v.n_layers, v.n_heads, v.input_size, config)
                weights = v.eject()
                self[k].control_trans = nn.Parameter(weights)

    def forward(self, batch_size):
        return {k: v(batch_size) for k, v in self.items()}


class PrefixTuningPool(nn.Module):
    """
    The model layer that holds all Prefix Tuning prefixes. While each Transformers layer has its own prefix, this layer
    is shared across all Transformers layers.

    How it works:

        1. A `PrefixTuningLayer` module that sets this module as pool module is added to each layer.
        2. On adding a prefix, each shim module where a prefix should be added increments a counter in `prefix_counts`.
        3. Finally, the base model class confirms adding a new prefix by calling `confirm_prefix()`.
        4. This module adds a prefix layer that produces outputs corresponding to the indicated number of layers.

    Notes:

        - The forward call to this layer is executed in the ForwardContext of each model pass.
        - All other methods of this class (except for `confirm_prefix()`) should be called exclusively by
          `PrefixTuningLayer`.

    Args:
        config (:class:`~transformers.PretrainedConfig`): The model config.
    """

    def __init__(self, model_config: PretrainedConfig, adapters_config: ModelAdaptersConfig):
        super().__init__()
        self.model_config = model_config
        self.adapters_config = adapters_config
        self.prefix_counts = {}
        self.prefix_tunings = nn.ModuleDict()

    def indicate_prefix(self, prefix_name: str, location_key: str, **kwargs):
        if prefix_name not in self.prefix_counts:
            self.prefix_counts[prefix_name] = {location_key: {"count": 1, **kwargs}}
        elif location_key not in self.prefix_counts[prefix_name]:
            self.prefix_counts[prefix_name][location_key] = {"count": 1, **kwargs}
        else:
            # TODO-AH: Check if kwargs are the same
            self.prefix_counts[prefix_name][location_key]["count"] += 1

        return self.prefix_counts[prefix_name][location_key]["count"] - 1

    def confirm_prefix(self, prefix_name: str) -> bool:
        """Create Prefix Tuning module based on shim layer infications."""
        prefix_tuning_config = self.adapters_config.match(prefix_name, PrefixTuningConfig)
        if prefix_tuning_config is None:
            return False

        if prefix_name not in self.prefix_counts:
            raise ValueError(f"Prefix {prefix_name} not found in PrefixTuningPool")

        module_configs = {}
        for location_key, location_config in self.prefix_counts[prefix_name].items():
            module_configs[location_key] = {
                "n_layers": location_config["count"],
                "n_heads": location_config["n_heads"],
                "input_size": location_config["input_size"],
                "n_embd_per_head": location_config["n_embd_per_head"],
            }
        prefix_tuning = PrefixTuningGroup(module_configs, prefix_tuning_config)
        prefix_tuning.train(self.training)  # make sure training mode is consistent
        self.prefix_tunings[prefix_name] = prefix_tuning
        del self.prefix_counts[prefix_name]
        return True

    def average_prefix(self, prefix_name: str, input_adapters: Dict[str, float]) -> bool:
        if self.confirm_prefix(prefix_name):
            # average weights
            avg_state_dict = {}
            for name, weight in input_adapters.items():
                module = self.prefix_tunings[name]
                if module is not None:
                    for k, v in module.state_dict().items():
                        if k in avg_state_dict:
                            avg_state_dict[k] += weight * v
                        else:
                            avg_state_dict[k] = weight * v
            # load averaged weights
            self.prefix_tunings[prefix_name].load_state_dict(avg_state_dict)
            return True

        return False

    def delete_prefix(self, prefix_name: str):
        if prefix_name in self.prefix_tunings:
            del self.prefix_tunings[prefix_name]

    def enable_prefix(self, prefix_name: str):
        if prefix_name in self.prefix_tunings:
            for param in self.prefix_tunings[prefix_name].parameters():
                param.requires_grad = True

    def get_prefix(self, prefix_name: str):
        if prefix_name in self.prefix_tunings:
            return self.prefix_tunings[prefix_name]
        else:
            return None

    def forward(self, *args, **kwargs):
        context = AdapterSetup.get_context()
        if context is not None:
            adapter_setup = context.adapter_setup
        else:
            adapter_setup = self.adapters_config.active_setup

        prefix_states = {}
        if adapter_setup is not None:
            # Infer batch size
            input_tensor_names = ["input_ids", "decoder_input_ids", "attention_mask", "inputs_embeds", "pixel_values"]
            batch_size = None
            for name in input_tensor_names:
                if kwargs.get(name, None) is not None:
                    batch_size = kwargs[name].size(0)
                    break
            if batch_size is None:
                if len(args) > 0:
                    batch_size = args[0].size(0)
                else:
                    raise ValueError("Could not infer batch size for prefix tuning from inputs.")

            # Pass to sub-layers
            for name in adapter_setup.flatten():
                if name in self.prefix_tunings:
                    prefix_states[name] = self.prefix_tunings[name](batch_size)

        return prefix_states


class PrefixTuningState(NamedTuple):
    """
    Models the input and output states of a prefix tuning layer.

    Args:
        key_states (torch.Tensor): The key states of the attention layer.
        value_states (torch.Tensor): The value states of the attention layer.
        residual_input (torch.Tensor): The residual input of the attention layer.
        attention_mask (torch.Tensor, optional): The attention mask of the attention layer.
        invert_mask (bool): Whether the attention mask is inverted (ie. using '1' for padding).
        idx_slice (slice, optional): Id slice for slicing prefix states along the batch size dimension.

    """

    key_states: torch.Tensor
    value_states: torch.Tensor
    residual_input: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    invert_mask: bool
    idx_slice: Optional[slice] = None


class PrefixTuningLayer(ComposableAdapterLayerBase, nn.Module):
    """
    Representation of a Prefix Tuning layer within one Transformer layer. This class implements `AdapterLayerBase` for
    compatibility with adapters. It uses `PrefixTuningPool` in the background and `set_pool()` must be called after
    initialization.

    Args:
        location_key (str): The id describing the location of this layer in the model.
                            Currently, can be "encoder_prefix", "cross_prefix" or None.
        config (:class:`~transformers.PretrainedConfig`): The model config.
    """

    adapter_modules_name = "prefixes"
    supported_compositions = [Stack, Parallel, BatchSplit]

    def __init__(
        self,
        location_key: str,
        model_config: PretrainedConfig,
        adapters_config: ModelAdaptersConfig,
        add_model_type_to_key: bool = False,
    ):
        super().__init__()
        self.model_config = model_config
        self.adapters_config = adapters_config
        self.location_key = location_key
        if add_model_type_to_key:
            self.location_key = f"{self.model_config.model_type}_{self.location_key}"
        self.prefixes = {}
        self.prefix_gates = nn.ModuleDict()

    def set_pool(self, pool: PrefixTuningPool):
        self.__setattr__("pool", pool)

    def add_adapter(self, adapter_name: str, layer_idx: int) -> bool:
        self.layer_idx = layer_idx
        # only match location keys for which we have config keys
        if self.location_key.startswith("cross") or self.location_key.startswith("encoder"):
            used_location_key = self.location_key
        else:
            used_location_key = None
        prefix_tuning_config = self.adapters_config.match(
            adapter_name,
            config_type=PrefixTuningConfig,
            layer_idx=self.layer_idx,
            location_key=used_location_key,
        )
        if prefix_tuning_config is not None:
            prefix_id = self.pool.indicate_prefix(
                adapter_name,
                self.location_key,
                n_heads=self.model_config.num_attention_heads,
                input_size=self.model_config.hidden_size,
                n_embd_per_head=getattr(self.model_config, "d_kv", None),  # this is currently specific to T5-3B
            )
            self.prefixes[adapter_name] = prefix_id

            if prefix_tuning_config.use_gating:
                gate_outputs = 1 if prefix_tuning_config.shared_gating else 2
                gate = nn.Linear(self.model_config.hidden_size, gate_outputs)
                gate.weight.data.normal_(mean=0.0, std=0.02)
                self.prefix_gates[adapter_name] = gate
            return True

        return False

    def average_adapter(self, adapter_name: str, input_adapters: Dict[str, float]) -> bool:
        # add new adapter
        if self.add_adapter(adapter_name, self.layer_idx):
            # prefix averaging is handled in pool, only average gates here
            if adapter_name in self.prefix_gates:
                avg_state_dict = {}
                for name, weight in input_adapters.items():
                    if name in self.prefix_gates:
                        module = self.prefix_gates[name]
                        for k, v in module.state_dict().items():
                            if k in avg_state_dict:
                                avg_state_dict[k] += weight * v
                            else:
                                avg_state_dict[k] = weight * v
                    else:
                        self.delete_adapter(adapter_name)  # clean up before raising error
                        raise ValueError("Adapter {} not found.".format(name))
                # load averaged weights
                self.prefix_gates[adapter_name].load_state_dict(avg_state_dict)
            return True
        else:
            return False

    def delete_adapter(self, adapter_name: str):
        self.pool.delete_prefix(adapter_name)
        if adapter_name in self.prefixes:
            del self.prefixes[adapter_name]
        if adapter_name in self.prefix_gates:
            del self.prefix_gates[adapter_name]

    def add_fusion_layer(self, adapter_names: Union[List, str]):
        pass  # not applicable to prefix tuning

    def delete_fusion_layer(self, adapter_names: Union[List, str]):
        pass  # not applicable to prefix tuning

    def enable_adapters(self, adapter_setup: AdapterCompositionBlock, unfreeze_adapters: bool, unfreeze_fusion: bool):
        if unfreeze_adapters:
            for prefix_tuning_name in adapter_setup.flatten():
                self.pool.enable_prefix(prefix_tuning_name)
                if prefix_tuning_name in self.prefix_gates:
                    for param in self.prefix_gates[prefix_tuning_name].parameters():
                        param.requires_grad = unfreeze_adapters

    def freeze_adapter(self, adapter_name: str, freeze: bool = True):
        if adapter_name in self.prefixes:
            self.pool.get_prefix(adapter_name)[self.location_key].train(not freeze)
            for param in self.pool.get_prefix(adapter_name)[self.location_key].parameters():
                param.requires_grad = not freeze
            if adapter_name in self.prefix_gates:
                for param in self.prefix_gates[adapter_name].parameters():
                    param.requires_grad = not freeze

    def get_adapter(self, adapter_name):
        return_dict = nn.ModuleDict()
        # Make sure to only return params once
        if adapter_name in self.prefixes and self.prefixes[adapter_name] == 0:
            prefix_module = self.pool.get_prefix(adapter_name)
            if prefix_module is not None:
                return_dict["prefix"] = prefix_module[self.location_key]
        if adapter_name in self.prefix_gates:
            return_dict["gate"] = self.prefix_gates[adapter_name]
        if len(return_dict) > 0:
            return return_dict

        return None

    def vslice(self, state: PrefixTuningState, slice_obj: slice) -> PrefixTuningState:
        if state.idx_slice is None:
            split_idx_slice = slice_obj
        else:
            split_idx_slice = slice(
                state.idx_slice.start + slice_obj.start,
                state.idx_slice.start + slice_obj.stop,
            )
        return PrefixTuningState(
            key_states=state.key_states[slice_obj],
            value_states=state.value_states[slice_obj],
            residual_input=state.residual_input[slice_obj],
            attention_mask=state.attention_mask[slice_obj] if state.attention_mask is not None else None,
            invert_mask=state.invert_mask,
            idx_slice=split_idx_slice,
        )

    def pad_and_concat(self, states: List[PrefixTuningState]) -> PrefixTuningState:
        """Pads all key & value states to the longest prefix length in the current batch.
        This is required e.g. for stacked prefix tunings.
        """
        max_prefix_length = max([state.key_states.shape[-2] for state in states])
        all_key_states, all_value_states, all_residual_input, all_attention_mask = [], [], [], []
        for state in states:
            key_states, value_states, residual_input, attention_mask = state[:4]
            # pad sizes
            pad_length = max_prefix_length - key_states.shape[-2]
            pad_size = (0, 0, pad_length, 0)
            key_states = F.pad(key_states, pad_size, "constant", self.model_config.pad_token_id)
            value_states = F.pad(value_states, pad_size, "constant", self.model_config.pad_token_id)

            # pad attention mask
            if pad_length > 0:
                # Masking the padded tokens only works correctly if attention_mask is set
                # We assume this to be the case at this point
                assert attention_mask is not None, "Attention mask must be set for prefix tuning"
                attention_mask = F.pad(
                    attention_mask,
                    (max_prefix_length - attention_mask.shape[-1], 0),
                    "constant",
                    1.0 if state.invert_mask else 0.0,
                )

            all_key_states.append(key_states)
            all_value_states.append(value_states)
            all_residual_input.append(residual_input)
            all_attention_mask.append(attention_mask)

        all_key_states = torch.cat(all_key_states, dim=0)
        all_value_states = torch.cat(all_value_states, dim=0)
        all_residual_input = torch.cat(all_residual_input, dim=0)
        all_attention_mask = torch.cat(all_attention_mask, dim=0) if attention_mask is not None else None

        return PrefixTuningState(
            key_states=all_key_states,
            value_states=all_value_states,
            residual_input=all_residual_input,
            attention_mask=all_attention_mask,
            invert_mask=states[0].invert_mask,
            idx_slice=states[0].idx_slice,
        )

    def repeat(self, state: PrefixTuningState, channels: int) -> PrefixTuningState:
        if state.attention_mask is not None:
            if state.attention_mask.dim() == 2:  # e.g. for DistilBERT, attention_mask has shape (batch_size, seq_len)
                attention_mask = state.attention_mask.repeat(channels, 1)
            else:
                attention_mask = state.attention_mask.repeat(channels, 1, 1, 1)
        else:
            attention_mask = None
        return PrefixTuningState(
            key_states=state.key_states.repeat(channels, 1, 1, 1),
            value_states=state.value_states.repeat(channels, 1, 1, 1),
            residual_input=state.residual_input.repeat(channels, 1, 1),
            attention_mask=attention_mask,
            invert_mask=state.invert_mask,
            idx_slice=state.idx_slice,
        )

    def mean(self, states: List[PrefixTuningState], weights: torch.Tensor) -> PrefixTuningState:
        # TODO implement average composition
        raise NotImplementedError()

    def compose_single(self, adapter_setup: str, state: PrefixTuningState, lvl: int = 0) -> PrefixTuningState:
        prefix_id = self.prefixes[adapter_setup]
        batch_size = state.key_states.size(0)

        # Retrieve pre-computed prefix states from context
        context = ForwardContext.get_context()
        # batch_size x n_heads x prefix_length x n_embd_per_head
        prefix_keys, prefix_values = context.prefix_states[adapter_setup][self.location_key][prefix_id]

        # Select index range for batch split
        # Ignore slices that go beyond the prefix states bsz
        # (this is the case for slices produced by Parallel blocks which operate on replicated kv states)
        if state.idx_slice is not None and state.idx_slice.start < prefix_keys.size(0):
            prefix_keys = prefix_keys[state.idx_slice]
            prefix_values = prefix_values[state.idx_slice]

        if adapter_setup in self.prefix_gates:
            gate = self.prefix_gates[adapter_setup]
            gate_output = torch.mean(torch.sigmoid(gate(state.residual_input)), dim=1)
            self._store_gating_score(adapter_setup, gate_output)
            gate_output_key = gate_output[:, 0].view(-1, 1, 1, 1)
            gate_output_value = gate_output[:, -1].view(-1, 1, 1, 1)
            prefix_keys = prefix_keys * gate_output_key
            prefix_values = prefix_values * gate_output_value

        # Replicate for Parallel block
        prefix_keys, prefix_values = adjust_tensors_for_parallel(state.key_states, prefix_keys, prefix_values)

        key_states = torch.cat([prefix_keys, state.key_states], dim=2)
        value_states = torch.cat([prefix_values, state.value_states], dim=2)
        if state.attention_mask is not None:
            if state.attention_mask.dim() == 2:  # e.g. for DistilBERT, attention_mask has shape (batch_size, seq_len)
                prefix_mask = torch.ones(batch_size, prefix_keys.size(2)).to(
                    device=state.attention_mask.device, dtype=state.attention_mask.dtype
                )
            else:
                prefix_mask = torch.ones(batch_size, 1, state.attention_mask.size(2), prefix_keys.size(2)).to(
                    device=state.attention_mask.device, dtype=state.attention_mask.dtype
                )
            if state.invert_mask:
                prefix_mask = 1.0 - prefix_mask
            (prefix_mask,) = adjust_tensors_for_parallel(state.attention_mask, prefix_mask)
            attention_mask = torch.cat([prefix_mask, state.attention_mask], dim=-1)
        else:
            attention_mask = None

        return state._replace(key_states=key_states, value_states=value_states, attention_mask=attention_mask)

    def forward(self, key_states, value_states, residual_input, attention_mask=None, invert_mask=True):
        adapter_setup = self.get_active_setup()
        if adapter_setup is not None:
            state = PrefixTuningState(key_states, value_states, residual_input, attention_mask, invert_mask)
            state = self.compose(adapter_setup, state)
            key_states, value_states, residual_input, attention_mask = state[:4]

        return key_states, value_states, attention_mask

import sys
from pathlib import Path

# Add the src directory to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2] / 'src'))

from typing import Dict, List, NamedTuple, Optional, Union
import os
import json
import torch
import torch.nn.functional as F
from torch import nn
import clip
from transformers import PretrainedConfig
from transformers.modeling_utils import ModuleUtilsMixin
from timm.models.vision_transformer import Block

# Absolute imports
from adapters.composition import AdapterCompositionBlock, BatchSplit, Parallel, Stack, adjust_tensors_for_parallel
from adapters.configuration import ModelAdaptersConfig, PrefixTuningConfig
from adapters.context import AdapterSetup, ForwardContext
from adapters.methods.adapter_layer_base import ComposableAdapterLayerBase
from adapters.methods.modeling import Activation_Function_Class
from transformers import LlamaConfig, LlamaTokenizer
from adapters import LlamaAdapterModel
from adapters.methods.utils import sample_top_p, _download

# Rest of your PrefixTuning classes and LLaMAPrefixTuning class implementation
# ...




class LLaMAPrefixTuning(nn.Module, ModuleUtilsMixin):
    def __init__(
        self,
        llama_ckpt_dir,
        llama_tokenizer,
        max_seq_len=512,
        max_batch_size=1,
        clip_model='ViT-L/14',
        v_embed_dim=768,
        v_depth=8,
        v_num_heads=16,
        v_mlp_ratio=4.0,
        query_len=10,
        query_layer=31,
        w_bias=False,
        w_lora=False,
        lora_rank=16,
        w_new_gate=False,
        phase="finetune",
        n_layers: int = None,
        n_heads: int = None,
        input_size: int = None,
        config: PrefixTuningConfig = None,
        n_embd_per_head: Optional[int] = None
    ):
        super().__init__()
        
        with open(os.path.join(llama_ckpt_dir, "params.json"), "r") as f:
            params = json.loads(f.read())
        w_bias = phase == "finetune"
        model_args: LlamaConfig = LlamaConfig(
            max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
        )

        self.clip, self.clip_transform = clip.load(clip_model)
        clip_dim = self.clip.visual.proj.shape[1]
        self.clip_proj = nn.Linear(clip_dim, v_embed_dim)
        self.clip_proj_norm = nn.LayerNorm(v_embed_dim)

        self.query_len = query_len
        self.query_layer = query_layer

        self.visual_query = nn.Embedding(query_len, v_embed_dim)
        self.visual_blocks = nn.ModuleList([
            Block(v_embed_dim, v_num_heads, v_mlp_ratio, qkv_bias=True)
            for _ in range(v_depth)
        ])
        self.visual_proj = nn.Linear(v_embed_dim, model_args.dim)
        self.visual_proj_norm = nn.LayerNorm(model_args.dim)

        self.adapter_query = nn.Embedding(
            query_len * query_layer, model_args.dim
        )

        self.tokenizer = LlamaTokenizer.from_pretrained(model_path=llama_tokenizer)

        model_args.w_bias = w_bias
        model_args.w_lora = w_lora
        model_args.lora_rank = lora_rank
        model_args.w_new_gate = w_new_gate
        model_args.vocab_size = self.tokenizer.n_words

        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        self.llama = LlamaAdapterModel(model_args)
        torch.set_default_tensor_type(torch.FloatTensor)

        ckpts = sorted(Path(llama_ckpt_dir).glob("*.pth"))
        for ckpt in ckpts:
            ckpt = torch.load(ckpt, map_location='cpu')
            self.llama.load_state_dict(ckpt, strict=False)

        del self.clip.transformer

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.phase = phase
        self.get_trainable_params(self.phase)

    def get_trainable_params(self, phase='finetune'):
        for name, para in self.named_parameters():
            para.requires_grad = False

        if phase == 'finetune':
            for name, para in self.named_parameters():
                if name.startswith("llama."):
                    if 'norm' in name or 'bias' in name:
                        para.data = para.data.float()
                        para.requires_grad = True
        elif phase == 'pretrain':
            train_param_name = [
                'gate', 'clip_proj', 'clip_proj_norm', 'visual_query',
                'visual_blocks', 'visual_proj', 'visual_proj_norm', 'adapter_query'
            ]
            for name, para in self.named_parameters():
                for train_name in train_param_name:
                    if train_name in name:
                        para.data = para.data.float()
                        para.requires_grad = True
        else:
            raise ValueError(f"Unknown model phase: {phase}")

    def clip_encode_image(self, x):
        x = self.clip.visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([
            self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
        ], dim=1)
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.clip.visual.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip.visual.ln_post(x[:, :, :])
        if self.clip.visual.proj is not None:
            x = x @ self.clip.visual.proj
        return x

    def forward_visual(self, imgs):
        clip_feats = self.clip_encode_image(imgs)
        clip_feats = self.clip_proj_norm(self.clip_proj(clip_feats.float()))
        visual_query = self.visual_query.weight.unsqueeze(0).repeat(len(imgs), 1, 1)
        visual_query = torch.cat([visual_query, clip_feats], dim=1)
        for block in self.visual_blocks:
            visual_query = block(visual_query)
        visual_query = visual_query[:, :self.query_len, :]
        visual_query = self.visual_proj(visual_query)
        visual_query = self.visual_proj_norm(visual_query)
        return visual_query

    def forward(self, tokens, labels, imgs):
        visual_query = self.forward_visual(imgs)
        _bsz, seqlen = tokens.shape
        h = self.llama.tok_embeddings(tokens)
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[:seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=0 + 1).type_as(h)
        for layer in self.llama.layers[:-1 * self.query_layer]:
            h = layer(h, 0, freqs_cis, mask)
        adapter = self.adapter_query.weight.reshape(self.query_layer, self.query_len, -1).unsqueeze(1)
        adapter_index = 0
        for layer in self.llama.layers[-1 * self.query_layer:]:
            dynamic_adapter = adapter[adapter_index].repeat(_bsz, 1, 1)
            dynamic_adapter = dynamic_adapter + visual_query
            h = layer(h, 0, freqs_cis, mask, dynamic_adapter)
            adapter_index += 1
        h = self.llama.norm(h)
        output = self.llama.output(h)
        output = output[:, :-1, :]
        labels = labels[:, 1:]
        if labels.sum() == 0:
            c_loss = output.mean() * 0
        else:
            c_loss = self.criterion(output.reshape(-1, self.llama.vocab_size), labels.flatten())
        return c_loss, c_loss

    @torch.inference_mode()
    def forward_inference(self, visual_query, tokens, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.llama.tok_embeddings(tokens)
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[start_pos: start_pos + seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)
        for layer in self.llama.layers[:-1 * self.query_layer]:
            h = layer(h, start_pos, freqs_cis, mask)
        adapter = self.adapter_query.weight.reshape(self.query_layer, self.query_len, -1).unsqueeze(1)
        adapter_index = 0
        for layer in self.llama.layers[-1 * self.query_layer:]:
            dynamic_adapter = adapter[adapter_index].repeat(_bsz, 1, 1)
            dynamic_adapter = dynamic_adapter + visual_query
            h = layer(h, start_pos, freqs_cis, mask, dynamic_adapter)
            adapter_index += 1
        h = self.llama.norm(h)
        output = self.llama.output(h[:, -1, :])
        return output.float()

    @torch.inference_mode()
    def generate(
        self, imgs, prompts,
        max_gen_len: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.75,
    ):
        bsz = len(imgs)
        params = self.llama.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        assert len(imgs) == len(prompts)

        with torch.cuda.amp.autocast():
            visual_query = self.forward_visual(imgs)

        if isinstance(prompts[0], str):
            prompts = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompts])
        max_prompt_size = max([len(t) for t in prompts])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()

        for k, t in enumerate(prompts):
            tokens[k, : len(t)] = torch.tensor(t).cuda().long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            with torch.cuda.amp.autocast():
                logits = self.forward_inference(visual_query, tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)

            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            if bsz == 1 and next_token[0] == self.tokenizer.eos_id:
                break
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):

            t = t[len(prompts[i]): len(prompts[i]) + max_gen_len]
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))

        return decoded

# Ensure this is added where needed in the module
class PrefixTuningGroup(nn.ModuleDict):
    def __init__(self, module_configs, prefix_tuning_config):
        super().__init__()
        if prefix_tuning_config["flat"]:
            prefix_tuning_class = FlatPrefixTuning
        else:
            prefix_tuning_class = PrefixTuning
        for k, kwargs in module_configs.items():
            self[k] = prefix_tuning_class(**kwargs, config=prefix_tuning_config)

    def eject(self):
        """Converts all PrefixTuning modules into FlatPrefixTuning modules."""
        for k, v in self.items():
            if isinstance(v, PrefixTuning):
                config = v.config.replace(flat=True)
                self[k] = FlatPrefixTuning(v.n_layers, v.n_heads, v.input_size, config)
                weights = v.eject()
                self[k].control_trans = nn.Parameter(weights)

    def forward(self, batch_size):
        return {k: v(batch_size) for k, v in self.items()}

class PrefixTuningPool(nn.Module):
    def __init__(self, model_config: PretrainedConfig, adapters_config: ModelAdaptersConfig):
        super().__init__()
        self.model_config = model_config
        self.adapters_config = adapters_config
        self.prefix_counts = {}
        self.prefix_tunings = nn.ModuleDict()

    def indicate_prefix(self, prefix_name: str, location_key: str, **kwargs):
        if prefix_name not in self.prefix_counts:
            self.prefix_counts[prefix_name] = {location_key: {"count": 1, **kwargs}}
        elif location_key not in self.prefix_counts[prefix_name]:
            self.prefix_counts[prefix_name][location_key] = {"count": 1, **kwargs}
        else:
            self.prefix_counts[prefix_name][location_key]["count"] += 1

        return self.prefix_counts[prefix_name][location_key]["count"] - 1

    def confirm_prefix(self, prefix_name: str) -> bool:
        prefix_tuning_config = self.adapters_config.match(prefix_name, PrefixTuningConfig)
        if prefix_tuning_config is None:
            return False

        if prefix_name not in self.prefix_counts:
            raise ValueError(f"Prefix {prefix_name} not found in PrefixTuningPool")

        module_configs = {}
        for location_key, location_config in self.prefix_counts[prefix_name].items():
            module_configs[location_key] = {
                "n_layers": location_config["count"],
                "n_heads": location_config["n_heads"],
                "input_size": location_config["input_size"],
                "n_embd_per_head": location_config["n_embd_per_head"],
            }
        prefix_tuning = PrefixTuningGroup(module_configs, prefix_tuning_config)
        prefix_tuning.train(self.training)
        self.prefix_tunings[prefix_name] = prefix_tuning
        del self.prefix_counts[prefix_name]
        return True

    def average_prefix(self, prefix_name: str, input_adapters: Dict[str, float]) -> bool:
        if self.confirm_prefix(prefix_name):
            avg_state_dict = {}
            for name, weight in input_adapters.items():
                module = self.prefix_tunings[name]
                if module is not None:
                    for k, v in module.state_dict().items():
                        if k in avg_state_dict:
                            avg_state_dict[k] += weight * v
                        else:
                            avg_state_dict[k] = weight * v
            self.prefix_tunings[prefix_name].load_state_dict(avg_state_dict)
            return True
        return False

    def delete_prefix(self, prefix_name: str):
        if prefix_name in self.prefix_tunings:
            del self.prefix_tunings[prefix_name]

    def enable_prefix(self, prefix_name: str):
        if prefix_name in self.prefix_tunings:
            for param in self.prefix_tunings[prefix_name].parameters():
                param.requires_grad = True

    def get_prefix(self, prefix_name: str):
        if prefix_name in self.prefix_tunings:
            return self.prefix_tunings[prefix_name]
        return None

    def forward(self, *args, **kwargs):
        context = AdapterSetup.get_context()
        if context is not None:
            adapter_setup = context.adapter_setup
        else:
            adapter_setup = self.adapters_config.active_setup

        prefix_states = {}
        if adapter_setup is not None:
            input_tensor_names = ["input_ids", "decoder_input_ids", "attention_mask", "inputs_embeds", "pixel_values"]
            batch_size = None
            for name in input_tensor_names:
                if kwargs.get(name, None) is not None:
                    batch_size = kwargs[name].size(0)
                    break
            if batch_size is None:
                if len(args) > 0:
                    batch_size = args[0].size(0)
                else:
                    raise ValueError("Could not infer batch size for prefix tuning from inputs.")

            for name in adapter_setup.flatten():
                if name in self.prefix_tunings:
                    prefix_states[name] = self.prefix_tunings[name](batch_size)

        return prefix_states

class PrefixTuningState(NamedTuple):
    key_states: torch.Tensor
    value_states: torch.Tensor
    residual_input: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    invert_mask: bool
    idx_slice: Optional[slice] = None

class PrefixTuningLayer(ComposableAdapterLayerBase, nn.Module):
    adapter_modules_name = "prefixes"
    supported_compositions = [Stack, Parallel, BatchSplit]

    def __init__(
        self,
        location_key: str,
        model_config: PretrainedConfig,
        adapters_config: ModelAdaptersConfig,
        add_model_type_to_key: bool = False,
    ):
        super().__init__()
        self.model_config = model_config
        self.adapters_config = adapters_config
        self.location_key = location_key
        if add_model_type_to_key:
            self.location_key = f"{self.model_config.model_type}_{self.location_key}"
        self.prefixes = {}
        self.prefix_gates = nn.ModuleDict()

    def set_pool(self, pool: PrefixTuningPool):
        self.__setattr__("pool", pool)

    def add_adapter(self, adapter_name: str, layer_idx: int) -> bool:
        self.layer_idx = layer_idx
        if self.location_key.startswith("cross") or self.location_key.startswith("encoder"):
            used_location_key = self.location_key
        else:
            used_location_key = None
        prefix_tuning_config = self.adapters_config.match(
            adapter_name,
            config_type=PrefixTuningConfig,
            layer_idx=self.layer_idx,
            location_key=used_location_key,
        )
        if prefix_tuning_config is not None:
            prefix_id = self.pool.indicate_prefix(
                adapter_name,
                self.location_key,
                n_heads=self.model_config.num_attention_heads,
                input_size=self.model_config.hidden_size,
                n_embd_per_head=getattr(self.model_config, "d_kv", None),
            )
            self.prefixes[adapter_name] = prefix_id

            if prefix_tuning_config.use_gating:
                gate_outputs = 1 if prefix_tuning_config.shared_gating else 2
                gate = nn.Linear(self.model_config.hidden_size, gate_outputs)
                gate.weight.data.normal_(mean=0.0, std=0.02)
                self.prefix_gates[adapter_name] = gate
            return True
        return False

    def average_adapter(self, adapter_name: str, input_adapters: Dict[str, float]) -> bool:
        if self.add_adapter(adapter_name, self.layer_idx):
            if adapter_name in self.prefix_gates:
                avg_state_dict = {}
                for name, weight in input_adapters.items():
                    if name in self.prefix_gates:
                        module = self.prefix_gates[name]
                        for k, v in module.state_dict().items():
                            if k in avg_state_dict:
                                avg_state_dict[k] += weight * v
                            else:
                                avg_state_dict[k] = weight * v
                    else:
                        self.delete_adapter(adapter_name)
                        raise ValueError("Adapter {} not found.".format(name))
                self.prefix_gates[adapter_name].load_state_dict(avg_state_dict)
            return True
        return False

    def delete_adapter(self, adapter_name: str):
        self.pool.delete_prefix(adapter_name)
        if adapter_name in self.prefixes:
            del self.prefixes[adapter_name]
        if adapter_name in self.prefix_gates:
            del self.prefix_gates[adapter_name]

    def add_fusion_layer(self, adapter_names: Union[List, str]):
        pass

    def delete_fusion_layer(self, adapter_names: Union[List, str]):
        pass

    def enable_adapters(self, adapter_setup: AdapterCompositionBlock, unfreeze_adapters: bool, unfreeze_fusion: bool):
        if unfreeze_adapters:
            for prefix_tuning_name in adapter_setup.flatten():
                self.pool.enable_prefix(prefix_tuning_name)
                if prefix_tuning_name in self.prefix_gates:
                    for param in self.prefix_gates[prefix_tuning_name].parameters():
                        param.requires_grad = unfreeze_adapters

    def freeze_adapter(self, adapter_name: str, freeze: bool = True):
        if adapter_name in self.prefixes:
            self.pool.get_prefix(adapter_name)[self.location_key].train(not freeze)
            for param in self.pool.get_prefix(adapter_name)[self.location_key].parameters():
                param.requires_grad = not freeze
            if adapter_name in self.prefix_gates:
                for param in self.prefix_gates[adapter_name].parameters():
                    param.requires_grad = not freeze

    def get_adapter(self, adapter_name):
        return_dict = nn.ModuleDict()
        if adapter_name in self.prefixes and self.prefixes[adapter_name] == 0:
            prefix_module = self.pool.get_prefix(adapter_name)
            if prefix_module is not None:
                return_dict["prefix"] = prefix_module[self.location_key]
        if adapter_name in self.prefix_gates:
            return_dict["gate"] = self.prefix_gates[adapter_name]
        if len(return_dict) > 0:
            return return_dict
        return None

    def vslice(self, state: PrefixTuningState, slice_obj: slice) -> PrefixTuningState:
        if state.idx_slice is None:
            split_idx_slice = slice_obj
        else:
            split_idx_slice = slice(
                state.idx_slice.start + slice_obj.start,
                state.idx_slice.start + slice_obj.stop,
            )
        return PrefixTuningState(
            key_states=state.key_states[slice_obj],
            value_states=state.value_states[slice_obj],
            residual_input=state.residual_input[slice_obj],
            attention_mask=state.attention_mask[slice_obj] if state.attention_mask is not None else None,
            invert_mask=state.invert_mask,
            idx_slice=split_idx_slice,
        )

    def pad_and_concat(self, states: List[PrefixTuningState]) -> PrefixTuningState:
        max_prefix_length = max([state.key_states.shape[-2] for state in states])
        all_key_states, all_value_states, all_residual_input, all_attention_mask = [], [], [], []
        for state in states:
            key_states, value_states, residual_input, attention_mask = state[:4]
            pad_length = max_prefix_length - key_states.shape[-2]
            pad_size = (0, 0, pad_length, 0)
            key_states = F.pad(key_states, pad_size, "constant", self.model_config.pad_token_id)
            value_states = F.pad(value_states, pad_size, "constant", self.model_config.pad_token_id)
            if pad_length > 0:
                assert attention_mask is not None, "Attention mask must be set for prefix tuning"
                attention_mask = F.pad(
                    attention_mask,
                    (max_prefix_length - attention_mask.shape[-1], 0),
                    "constant",
                    1.0 if state.invert_mask else 0.0,
                )
            all_key_states.append(key_states)
            all_value_states.append(value_states)
            all_residual_input.append(residual_input)
            all_attention_mask.append(attention_mask)

        all_key_states = torch.cat(all_key_states, dim=0)
        all_value_states = torch.cat(all_value_states, dim=0)
        all_residual_input = torch.cat(all_residual_input, dim=0)
        all_attention_mask = torch.cat(all_attention_mask, dim=0) if attention_mask is not None else None
        return PrefixTuningState(
            key_states=all_key_states,
            value_states=all_value_states,
            residual_input=all_residual_input,
            attention_mask=all_attention_mask,
            invert_mask=states[0].invert_mask,
            idx_slice=states[0].idx_slice,
        )

    def repeat(self, state: PrefixTuningState, channels: int) -> PrefixTuningState:
        if state.attention_mask is not None:
            if state.attention_mask.dim() == 2:
                attention_mask = state.attention_mask.repeat(channels, 1)
            else:
                attention_mask = state.attention_mask.repeat(channels, 1, 1, 1)
        else:
            attention_mask = None
        return PrefixTuningState(
            key_states=state.key_states.repeat(channels, 1, 1, 1),
            value_states=state.value_states.repeat(channels, 1, 1, 1),
            residual_input=state.residual_input.repeat(channels, 1, 1),
            attention_mask=attention_mask,
            invert_mask=state.invert_mask,
            idx_slice=state.idx_slice,
        )

    def mean(self, states: List[PrefixTuningState], weights: torch.Tensor) -> PrefixTuningState:
        raise NotImplementedError()

    def compose_single(self, adapter_setup: str, state: PrefixTuningState, lvl: int = 0) -> PrefixTuningState:
        prefix_id = self.prefixes[adapter_setup]
        batch_size = state.key_states.size(0)
        context = ForwardContext.get_context()
        prefix_keys, prefix_values = context.prefix_states[adapter_setup][self.location_key][prefix_id]
        if state.idx_slice is not None and state.idx_slice.start < prefix_keys.size(0):
            prefix_keys = prefix_keys[state.idx_slice]
            prefix_values = prefix_values[state.idx_slice]
        if adapter_setup in self.prefix_gates:
            gate = self.prefix_gates[adapter_setup]
            gate_output = torch.mean(torch.sigmoid(gate(state.residual_input)), dim=1)
            self._store_gating_score(adapter_setup, gate_output)
            gate_output_key = gate_output[:, 0].view(-1, 1, 1, 1)
            gate_output_value = gate_output[:, -1].view(-1, 1, 1, 1)
            prefix_keys = prefix_keys * gate_output_key
            prefix_values = prefix_values * gate_output_value
        prefix_keys, prefix_values = adjust_tensors_for_parallel(state.key_states, prefix_keys, prefix_values)
        key_states = torch.cat([prefix_keys, state.key_states], dim=2)
        value_states = torch.cat([prefix_values, state.value_states], dim=2)
        if state.attention_mask is not None:
            if state.attention_mask.dim() == 2:
                prefix_mask = torch.ones(batch_size, prefix_keys.size(2)).to(
                    device=state.attention_mask.device, dtype=state.attention_mask.dtype
                )
            else:
                prefix_mask = torch.ones(batch_size, 1, state.attention_mask.size(2), prefix_keys.size(2)).to(
                    device=state.attention_mask.device, dtype=state.attention_mask.dtype
                )
            if state.invert_mask:
                prefix_mask = 1.0 - prefix_mask
            (prefix_mask,) = adjust_tensors_for_parallel(state.attention_mask, prefix_mask)
            attention_mask = torch.cat([prefix_mask, state.attention_mask], dim=-1)
        else:
            attention_mask = None
        return state._replace(key_states=key_states, value_states=value_states, attention_mask=attention_mask)

    def forward(self, key_states, value_states, residual_input, attention_mask=None, invert_mask=True):
        adapter_setup = self.get_active_setup()
        if adapter_setup is not None:
            state = PrefixTuningState(key_states, value_states, residual_input, attention_mask, invert_mask)
            state = self.compose(adapter_setup, state)
            key_states, value_states, residual_input, attention_mask = state[:4]
        return key_states, value_states, attention_mask


def Llama7B_adapter(args, **kwargs):
    llama_model_path = args.llama_model_path
    model_name = "7B"
    checkpoint = torch.load(llama_model_path + model_name + "/consolidated.00.pth", map_location="cpu")
    with open(llama_model_path + model_name + "/params.json", "r") as f:
        params = json.loads(f.read())

    model_args: LlamaConfig = LlamaConfig(
        max_seq_len=args.max_seq_len,
        max_batch_size=32,
        adapter_len=args.adapter_len,
        adapter_layer=args.adapter_layer,
        **params
    )
    tokenizer = LlamaTokenizer.from_pretrained(model_path=llama_model_path + "/tokenizer.model")
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model_llama_adapter = LlamaAdapterModel.from_pretrained(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model_llama_adapter.load_state_dict(checkpoint, strict=False)

    for name, param in model_llama_adapter.named_parameters():
        if "adapter" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            param.data = param.data.float()

    for name, param in model_llama_adapter.layers[-1 * args.adapter_layer :].named_parameters():
        if "gate" in name or "adapter" in name:
            param.data = param.data.float()
            param.requires_grad = True

    return model_llama_adapter