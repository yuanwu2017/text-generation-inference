# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.

import bisect
from dataclasses import dataclass
from functools import wraps
import itertools
import math
import os
import tempfile
import time
import copy
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch._dynamo
from loguru import logger
from opentelemetry import trace

import text_generation_server.habana_quantization_env as hq_env
import habana_frameworks.torch as htorch
from optimum.habana.utils import HabanaProfile
from optimum.habana.transformers.generation import MODELS_OPTIMIZED_WITH_STATIC_SHAPES
from text_generation_server.utils.chunks import concat_text_chunks
from text_generation_server.utils.speculate import get_speculate
from optimum.habana.checkpoint_utils import (
    get_repo_root,
    model_on_meta,
    write_checkpoints_json,
)
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    AutoConfig,
)

from text_generation_server.utils.tokens import batch_top_tokens
from text_generation_server.models import Model
from text_generation_server.utils.chunks import concat_text_chunks
from text_generation_server.utils.import_utils import SYSTEM
from text_generation_server.utils.quantization import get_loader
from text_generation_server.utils.tokens import batch_top_tokens
from text_generation_server.models.types import (
    Batch,
    Tokens,
    Generation,
    GeneratedText,
)
from text_generation_server.pb import generate_pb2
from text_generation_server.utils import NextTokenChooser, StoppingCriteria, Sampling
from text_generation_server.utils.debug import dbg_trace
from optimum.habana.utils import get_hpu_memory_stats
from text_generation_server.utils.debug import dbg_trace

tracer = trace.get_tracer(__name__)
MAX_TOTAL_TOKENS = int(os.environ.get("MAX_TOTAL_TOKENS", 2048))
PAD_SEQUENCE_TO_MULTIPLE_OF = int(os.environ.get("PAD_SEQUENCE_TO_MULTIPLE_OF", 256))
CHUNK_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
LAZY_MODE = int(os.environ.get("PT_HPU_LAZY_MODE", 1))
BATCH_SIZE_EXPONENT_BASE = int(os.environ.get("BATCH_SIZE_EXPONENT_BASE", 2))
MAX_BATCH_SIZE = (
    int(os.environ.get("MAX_BATCH_SIZE"))
    if os.environ.get("MAX_BATCH_SIZE") is not None
    else None
)


def torch_compile_for_eager(func):
    if LAZY_MODE == 1:
        return func
    return torch.compile(
        func, backend="hpu_backend", options={"keep_input_mutations": True}
    )


def round_up_seq(number, k):
    return (number + k - 1) // k * k


def round_up_batch(number):
    return BATCH_SIZE_EXPONENT_BASE ** (
        math.ceil(math.log(number, BATCH_SIZE_EXPONENT_BASE))
    )



def remove_kv_cache_from_output(module):
    orig_fwd = module.forward

    @wraps(orig_fwd)
    def forward(*args, **kwargs):
        if kwargs["past_key_values"] is not None:
            kwargs["return_dict"] = False
            output = orig_fwd(*args, **kwargs)
            first_value, second_value, *_ = output
            if first_value.nelement() < 2:
                return second_value
            else:
                return first_value
        else:
            kwargs["return_dict"] = True
            return orig_fwd(*args, **kwargs)

    module.forward = forward
    return module




@dataclass
class CausalLMBatch(Batch):
    batch_id: int
    requests: List[generate_pb2.Request]
    requests_idx_mapping: Dict[int, int]

    # Decoder values
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    past_key_values: Optional[List[Tuple]]

    # All tokens
    all_input_ids: List[torch.Tensor]

    # Lengths of all generations present in the batch
    input_lengths: List[int]
    prefix_offsets: List[int]
    read_offsets: List[int]

    # Generation helpers
    next_token_choosers: List[NextTokenChooser]
    stopping_criterias: List[StoppingCriteria]
    top_n_tokens: List[int]
    top_n_tokens_tensor: torch.Tensor

    # Metadata used for padding
    max_input_length: int
    padding_right_offset: int

    # Maximum number of tokens this batch will grow to
    max_tokens: int

    # Past metadata
    keys_head_dim_last: bool = True

    def to_pb(self) -> generate_pb2.CachedBatch:
        return generate_pb2.CachedBatch(
            id=self.batch_id,
            request_ids=[r.id for r in self.requests],
            size=len(self),
            max_tokens=self.max_tokens,
            current_tokens=len(self.input_ids),
        )

    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "CausalLMBatch":
        inputs = []
        next_token_choosers = []
        stopping_criterias = []
        top_n_tokens = []
        prefix_offsets = []
        read_offsets = []
        requests_idx_mapping = {}

        # Parse batch
        max_truncation = 0
        padding_right_offset = 0
        max_decode_tokens = 0
        for i, r in enumerate(pb.requests):
            requests_idx_mapping[r.id] = i
            inputs.append(concat_text_chunks(r.input_chunks.chunks))

            next_token_choosers.append(
                NextTokenChooser.from_pb(r.parameters, device, tokenizer)
            )
            stopping_criteria = StoppingCriteria.from_pb(
                r.stopping_parameters, tokenizer
            )
            stopping_criterias.append(stopping_criteria)
            top_n_tokens.append(r.top_n_tokens)
            max_truncation = max(max_truncation, r.truncate)
            max_decode_tokens += stopping_criteria.max_new_tokens
            padding_right_offset = max(
                padding_right_offset, stopping_criteria.max_new_tokens
            )

        tokenized_inputs = tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
            truncation=True,
            max_length=max_truncation,
        ).to(device)
        for _ in pb.requests:
            input_len = tokenized_inputs["input_ids"].shape[1]
            prefix_offsets.append(input_len - 5)
            read_offsets.append(input_len)

        input_lengths = tokenized_inputs["attention_mask"].sum(1)
        max_input_length = input_lengths.max()

        input_ids = tokenized_inputs["input_ids"]
        # Allocate maximum attention_mask
        attention_mask = input_ids.new_zeros(
            (pb.size, max_input_length + padding_right_offset)
        )
        # Copy tokenizer attention_mask into fully allocated attention_mask
        attention_mask[:, :max_input_length] = tokenized_inputs["attention_mask"]

        position_ids = tokenized_inputs["attention_mask"].long().cumsum(-1) - 1
        position_ids.masked_fill_(tokenized_inputs["attention_mask"] == 0, 1)
        all_input_ids = tokenized_inputs["input_ids"].T.split(1, dim=1)
        top_n_tokens_tensor = torch.tensor(
            top_n_tokens, device=device, dtype=torch.int64
        )

        max_tokens = len(inputs) * (max_input_length + max_decode_tokens)

        return cls(
            batch_id=pb.id,
            requests=pb.requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            all_input_ids=list(all_input_ids),
            input_lengths=input_lengths.tolist(),
            prefix_offsets=prefix_offsets,
            read_offsets=read_offsets,
            next_token_choosers=next_token_choosers,
            stopping_criterias=stopping_criterias,
            top_n_tokens=top_n_tokens,
            top_n_tokens_tensor=top_n_tokens_tensor,
            max_input_length=max_input_length.item(),
            padding_right_offset=padding_right_offset,
            max_tokens=max_tokens,
        )

    @tracer.start_as_current_span("filter")
    def filter(self, request_ids: List[int]) -> Optional["CausalLMBatch"]:
        if len(request_ids) == 0:
            raise ValueError("Batch must have at least one request")
        if len(request_ids) == len(self):
            return self

        keep_indices = []

        # New values after filtering
        requests_idx_mapping = {}
        requests = []
        input_lengths = []
        prefix_offsets = []
        read_offsets = []
        all_input_ids = []
        max_input_length = 0

        next_token_choosers = []
        stopping_criterias = []
        top_n_tokens = []

        total_remaining_decode_tokens = 0
        new_padding_right_offset = 0

        for i, request_id in enumerate(request_ids):
            idx = self.requests_idx_mapping[request_id]
            requests_idx_mapping[request_id] = i
            keep_indices.append(idx)

            requests.append(self.requests[idx])
            prefix_offsets.append(self.prefix_offsets[idx])
            read_offsets.append(self.read_offsets[idx])
            all_input_ids.append(self.all_input_ids[idx])

            request_input_length = self.input_lengths[idx]
            input_lengths.append(request_input_length)
            max_input_length = max(max_input_length, request_input_length)

            next_token_choosers.append(self.next_token_choosers[idx])
            stopping_criteria = self.stopping_criterias[idx]
            stopping_criterias.append(stopping_criteria)
            top_n_tokens.append(self.top_n_tokens[idx])
            remaining_decode_tokens = (
                stopping_criteria.max_new_tokens - stopping_criteria.current_tokens
            )
            total_remaining_decode_tokens += remaining_decode_tokens
            new_padding_right_offset = max(
                new_padding_right_offset, remaining_decode_tokens
            )

        # Apply indices to input_ids, attention mask, past key values and other items that need to be cached
        input_ids = self.input_ids[keep_indices]
        position_ids = self.position_ids[keep_indices]
        self.attention_mask = self.attention_mask[
            keep_indices,
            -(self.padding_right_offset + max_input_length) : (
                self.attention_mask.shape[1] - self.padding_right_offset
            )
            + new_padding_right_offset,
        ]

        # Ensure that past_key_values tensors can be updated in-place
        if type(self.past_key_values[0]) is tuple:
            self.past_key_values = [list(layer) for layer in self.past_key_values]

        # Update tensors in-place to allow incremental garbage collection
        past_kv_length = max_input_length - 1
        for layer in self.past_key_values:
            past_keys, past_values = layer
            if len(past_keys.shape) == 3:
                # Force past to be of dim [self_size, num_heads, ...] for easy indexing
                past_keys = past_keys.view(len(self), -1, *past_keys.shape[-2:])
                past_values = past_values.view(len(self), -1, *past_values.shape[-2:])
            if self.keys_head_dim_last:
                layer[0] = past_keys[keep_indices, :, -past_kv_length:, :]
            else:
                layer[0] = past_keys[keep_indices, :, :, -past_kv_length:]
            del past_keys
            layer[1] = past_values[keep_indices, :, -past_kv_length:, :]
            del past_values

        top_n_tokens_tensor = self.top_n_tokens_tensor[keep_indices]
        max_tokens = len(request_ids) * max_input_length + total_remaining_decode_tokens

        self.requests = requests
        self.requests_idx_mapping = requests_idx_mapping
        self.input_ids = input_ids
        self.position_ids = position_ids
        self.all_input_ids = all_input_ids
        self.input_lengths = input_lengths
        self.prefix_offsets = prefix_offsets
        self.read_offsets = read_offsets
        self.next_token_choosers = next_token_choosers
        self.stopping_criterias = stopping_criterias
        self.top_n_tokens = top_n_tokens
        self.top_n_tokens_tensor = top_n_tokens_tensor
        self.max_input_length = max_input_length
        self.padding_right_offset = new_padding_right_offset
        self.max_tokens = max_tokens

        return self

    @classmethod
    @tracer.start_as_current_span("concatenate")
    def concatenate(cls, batches: List["CausalLMBatch"]) -> "CausalLMBatch":
        # Used for padding
        total_batch_size = 0
        max_input_length = 0
        padding_right_offset = 0
        for batch in batches:
            total_batch_size += len(batch)
            max_input_length = max(max_input_length, batch.max_input_length)
            padding_right_offset = max(padding_right_offset, batch.padding_right_offset)

        # Batch attributes
        requests = []
        requests_idx_mapping = {}
        input_lengths = []
        prefix_offsets = []
        read_offsets = []
        all_input_ids = []
        next_token_choosers = []
        stopping_criterias = []
        top_n_tokens = []
        max_tokens = 0

        # Batch tensors
        input_ids = None
        attention_mask = None
        position_ids = None
        past_key_values = []
        top_n_tokens_tensor = None

        # Used for slicing correctly inside the tensors
        # Equivalent to a cumsum on batch sizes
        start_index = 0
        for i, batch in enumerate(batches):
            requests.extend(batch.requests)
            input_lengths.extend(batch.input_lengths)
            prefix_offsets.extend(batch.prefix_offsets)
            read_offsets.extend(batch.read_offsets)
            all_input_ids.extend(batch.all_input_ids)
            next_token_choosers.extend(batch.next_token_choosers)
            stopping_criterias.extend(batch.stopping_criterias)
            top_n_tokens.extend(batch.top_n_tokens)

            if i == 0:
                requests_idx_mapping = batch.requests_idx_mapping
            else:
                # We need to offset the mapping for each batch by the cumulative batch size
                for k, v in batch.requests_idx_mapping.items():
                    requests_idx_mapping[k] = v + start_index

            # Slicing end index for this batch
            end_index = start_index + len(batch)

            # We only concatenate batches that did at least one step
            if batch.past_key_values is None:
                raise ValueError("only concatenate prefilled batches")

            # Create empty tensor
            # input_ids is always of shape [batch_size, 1]
            # We do not need to pad it
            if input_ids is None:
                input_ids = batch.input_ids.new_empty((total_batch_size, 1))
            # Copy to correct indices
            input_ids[start_index:end_index] = batch.input_ids

            # Create padded tensor
            if attention_mask is None:
                attention_mask = batch.attention_mask.new_zeros(
                    (total_batch_size, max_input_length + padding_right_offset),
                )

            if top_n_tokens_tensor is None:
                top_n_tokens_tensor = batches[0].top_n_tokens_tensor.new_zeros(
                    total_batch_size,
                )
            top_n_tokens_tensor[start_index:end_index] = batch.top_n_tokens_tensor

            # We need to slice the attention mask to remove padding from previous steps
            # and to remove unused allocated space
            left_offset = max_input_length - batch.max_input_length
            batch_left_offset = (
                batch.attention_mask.shape[1]
                - batch.max_input_length
                - batch.padding_right_offset
            )
            attention_mask[
                start_index:end_index,
                left_offset:-padding_right_offset,
            ] = batch.attention_mask[
                :,
                batch_left_offset : -batch.padding_right_offset,
            ]

            # Create empty tensor
            # position_ids is always of shape [batch_size, 1]
            if position_ids is None:
                position_ids = batch.position_ids.new_empty((total_batch_size, 1))
            position_ids[start_index:end_index] = batch.position_ids

            # Shenanigans to get dimensions because BLOOM outputs a past with a different shape
            # BLOOM Keys:   [batch_size * num_heads, head_dim, seq_length]
            # BLOOM Values: [batch_size * num_heads, seq_length, head_dim]
            # And ensure that we can update tensors in-place
            if isinstance(batch.past_key_values[0], tuple):
                batch.past_key_values = [
                    [t.view(len(batch), -1, *t.shape[-2:]) for t in layer]
                    for layer in batch.past_key_values
                ]
            elif len(batch.past_key_values[0][0].shape) == 3:
                for layer in batch.past_key_values:
                    for k, t in enumerate(layer):
                        layer[k] = t.view(len(batch), -1, *t.shape[-2:])

            # Add eventual padding tokens that were added while concatenating
            max_tokens += batch.max_tokens + (
                max_input_length - batch.max_input_length
            ) * len(batch)

            start_index = end_index

        first_past_kvs = batches[0].past_key_values
        _, num_heads, padded_sequence_length, head_dim = first_past_kvs[0][1].shape

        padded_past_values_shape = (
            total_batch_size,
            num_heads,
            max_input_length - 1,
            head_dim,
        )

        if batches[0].keys_head_dim_last:
            padded_past_keys_shape = padded_past_values_shape
        else:
            # seq_length is last for BLOOM
            padded_past_keys_shape = (
                total_batch_size,
                num_heads,
                head_dim,
                max_input_length - 1,
            )

        # Iterate over attention layers
        # Concatenate past key values layer by layer to allow incremental garbage collection
        for j in range(len(first_past_kvs)):
            padded_past_keys = first_past_kvs[j][0].new_zeros(padded_past_keys_shape)
            start_index = 0
            for batch in batches:
                past_keys = batch.past_key_values[j][0]
                # Clear reference to the original tensor
                batch.past_key_values[j][0] = None

                # Slicing end index for this batch
                end_index = start_index + len(batch)
                # We slice the keys to remove the padding from previous batches
                past_seq_len = batch.max_input_length - 1
                if batch.keys_head_dim_last:
                    padded_past_keys[start_index:end_index, :, -past_seq_len:, :] = (
                        past_keys[:, :, -past_seq_len:, :]
                    )
                else:
                    # BLOOM case
                    padded_past_keys[start_index:end_index, :, :, -past_seq_len:] = (
                        past_keys[:, :, :, -past_seq_len:]
                    )
                del past_keys

                start_index = end_index

            padded_past_values = first_past_kvs[j][1].new_zeros(
                padded_past_values_shape
            )
            start_index = 0
            for batch in batches:
                past_values = batch.past_key_values[j][1]
                # Clear reference to the original tensor
                batch.past_key_values[j][1] = None

                # Slicing end index for this batch
                end_index = start_index + len(batch)
                # We slice the past values to remove the padding from previous batches
                past_seq_len = batch.max_input_length - 1
                padded_past_values[start_index:end_index, :, -past_seq_len:, :] = (
                    past_values[:, :, -past_seq_len:, :]
                )
                del past_values

                # Update values
                start_index = end_index

            past_key_values.append([padded_past_keys, padded_past_values])

        return cls(
            batch_id=batches[0].batch_id,
            requests=requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            all_input_ids=all_input_ids,
            input_lengths=input_lengths,
            prefix_offsets=prefix_offsets,
            read_offsets=read_offsets,
            next_token_choosers=next_token_choosers,
            stopping_criterias=stopping_criterias,
            top_n_tokens=top_n_tokens,
            top_n_tokens_tensor=top_n_tokens_tensor,
            max_input_length=max_input_length,
            padding_right_offset=padding_right_offset,
            keys_head_dim_last=batches[0].keys_head_dim_last,
            max_tokens=max_tokens,
        )

    def __len__(self):
        return len(self.requests)


@dataclass
class CausalLMBatchKeysLast(CausalLMBatch):
    keys_head_dim_last: bool = False


class CausalLM(Model):
    def __init__(
        self,
        model_id: str,
        model_class: Optional[Type[torch.nn.Module]] = None,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        speculator: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        default_dtype=torch.float16,
        trust_remote_code: bool = False,
        tokenizer_class=AutoTokenizer,
        config_class=AutoConfig,
        batch_class=CausalLMBatch,
    ):
        if speculator:
            raise RuntimeError("Speculator decoding is not enabled for AutoModel")

        self.prev_bs = 0
        self.quantize = quantize

        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )

        # Create model
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        rank = int(os.getenv("RANK", "0"))
        dtype = torch.bfloat16 if dtype is None else dtype
        device = torch.device("hpu")

        if hq_env.is_quantization_enabled:
            htorch.core.hpu_set_env()

        if world_size > 1:
            os.environ.setdefault(
                "DEEPSPEED_USE_HABANA_FRAMEWORKS_DETERMINISTIC_API", "1"
            )
            model = self.get_deepspeed_model(model_id, dtype, revision)
            model = hq_env.prepare_model_for_quantization(model)
        else:
            get_repo_root(model_id)

            # Check support for rope scaling
            model_kwargs = {}
            config = AutoConfig.from_pretrained(model_id)
            if hasattr(config, "rope_scaling"):
                model_kwargs["rope_scaling"] = self.get_rope_scaling()

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                revision=revision,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
                **model_kwargs,
            )
            model = hq_env.prepare_model_for_quantization(model)
            model = model.eval().to(device)

        self.enable_hpu_graph = (
            os.getenv("ENABLE_HPU_GRAPH", "true").lower() == "true" and LAZY_MODE == 1
        )
        self.limit_hpu_graph = os.getenv("LIMIT_HPU_GRAPH", "true").lower() == "true"

        if model.config.model_type not in [
            "gpt_bigcode"
        ]:  # gpt_bigcode/starcoderbase-3b skips remove_kv_cache_from_output()
            model = remove_kv_cache_from_output(model)

        if self.enable_hpu_graph:
            from habana_frameworks.torch.hpu import wrap_in_hpu_graph

            model = wrap_in_hpu_graph(model, disable_tensor_cache=True)
        else:
            if LAZY_MODE == 0:
                # It is said that "keep_input_mutations" is safe for inference to be done
                dbg_trace("TORCH COMPILE", "Torch compiling of model")
                model.model = torch.compile(
                    model.model,
                    backend="hpu_backend",
                    options={"keep_input_mutations": True},
                )

        model = hq_env.setup_quantization(model)

        if model.config.model_type not in MODELS_OPTIMIZED_WITH_STATIC_SHAPES:
            raise ValueError(f"Model type {model.config.model_type} is not supported!")

        if tokenizer.pad_token_id is None:
            if model.config.pad_token_id is not None:
                tokenizer.pad_token_id = model.config.pad_token_id
            elif model.config.eos_token_id is not None:
                if isinstance(model.config.eos_token_id, int):
                    tokenizer.pad_token_id = model.config.eos_token_id
                elif isinstance(model.config.eos_token_id, list):
                    tokenizer.pad_token_id = model.config.eos_token_id[0]
                else:
                    raise ValueError(
                        f"{type(model.config.eos_token_id)} type of eos_token_id in the model's config is not supported for tokenizer.pad_token_id"
                    )
            elif tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.kwargs = {
            "use_cache": True,
            "return_dict": True,
        }

        if model.config.model_type in [
            "llama",
            "mistral",
            "starcoder2",
            "qwen2",
            "falcon",
            "gpt_bigcode",
        ]:
            if model.config.model_type not in ["falcon", "gpt_bigcode"]:
                self.kwargs["attn_softmax_bf16"] = True

            if model.config.model_type not in ["gpt_bigcode"]:
                self.kwargs["trim_logits"] = True

            if os.getenv("USE_FLASH_ATTENTION", "true").lower() == "true":
                self.kwargs["use_flash_attention"] = True
            if os.getenv("FLASH_ATTENTION_RECOMPUTE", "true").lower() == "true":
                self.kwargs["flash_attention_recompute"] = True

        self.speculate = get_speculate()
        self.batch_class = CausalLMBatch

        super(CausalLM, self).__init__(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            requires_padding=True,
            dtype=dtype,
            device=device,
            rank=rank,
        )

        # Create profiler
        ranks_to_profile = [int(val) for val in os.getenv("PROF_RANKS", "0").split(",")]
        record_shapes = os.getenv("PROF_RECORD_SHAPES", "false").lower() == "true"
        output_dir = os.getenv("PROF_PATH", "/tmp/hpu_profile")
        self.profiling_warmup_steps = (
            int(os.getenv("PROF_WARMUPSTEP", "0")) if rank in ranks_to_profile else 0
        )
        self.profiling_steps = (
            int(os.getenv("PROF_STEP", "0")) if rank in ranks_to_profile else 0
        )
        self.profiling_wait_steps = int(os.getenv("PROF_WAITSTEP", "0"))
        if self.profiling_steps > 0:
            self.hb_profiler = HabanaProfile(
                wait=self.profiling_wait_steps,
                warmup=self.profiling_warmup_steps,
                active=self.profiling_steps,
                output_dir=output_dir,
                record_shapes=record_shapes,
            )
            self.hb_profiler.start()
        else:
            self.hb_profiler = None
        self.step = 0

    def get_deepspeed_model(
        self, model_id: str, dtype: torch.dtype, revision: Optional[str] = None
    ) -> torch.nn.Module:
        import deepspeed
        from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu

        world_size, rank, local_rank = initialize_distributed_hpu()
        model_kwargs = {"revision": revision}

        # Initialize process(es) for DeepSpeed
        deepspeed.init_distributed(dist_backend="hccl")
        logger.info(
            "DeepSpeed is enabled. world_size {} rank {} local_rank {}".format(
                world_size, rank, local_rank
            )
        )
        config = AutoConfig.from_pretrained(model_id, **model_kwargs)
        load_to_meta = model_on_meta(config)

        # Check support for rope scaling
        if hasattr(config, "rope_scaling"):
            config.rope_scaling = self.get_rope_scaling()
            model_kwargs["rope_scaling"] = self.get_rope_scaling()

        if load_to_meta:
            # Construct model with fake meta tensors, later will be replaced on devices during ds-inference ckpt load
            with deepspeed.OnDevice(dtype=dtype, device="meta"):
                model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
        else:
            get_repo_root(model_id, local_rank=os.getenv("LOCAL_RANK"))
            # TODO: revisit placement on CPU when auto-injection is possible
            with deepspeed.OnDevice(dtype=dtype, device="cpu"):
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, torch_dtype=dtype, **model_kwargs
                )
        model = model.eval()

        # Initialize the model
        ds_inference_kwargs = {"dtype": dtype}
        ds_inference_kwargs["tensor_parallel"] = {"tp_size": world_size}
        ds_inference_kwargs["enable_cuda_graph"] = False

        if load_to_meta:
            # model loaded to meta is managed differently
            checkpoints_json = tempfile.NamedTemporaryFile(suffix=".json", mode="+w")
            write_checkpoints_json(model_id, local_rank, checkpoints_json)
            ds_inference_kwargs["checkpoint"] = checkpoints_json.name
        model = deepspeed.init_inference(model, **ds_inference_kwargs)

        return model.module

    def get_rope_scaling(self) -> Optional[Dict]:
        rope_scaling = os.getenv("ROPE_SCALING", None)
        if rope_scaling is None:
            return None

        rope_factor = float(os.getenv("ROPE_FACTOR", 1.0))
        return {"type": rope_scaling, "factor": float(rope_factor)}


    @classmethod
    def fallback(
        cls,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        speculator: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ):
        if speculator:
            raise RuntimeError("Speculator decoding is not enabled for AutoModel")

        device_count = 0
        if torch.cuda.is_available():
            device = torch.device("cuda")
            device_count = torch.cuda.device_count()
            dtype = torch.float16 if dtype is None else dtype
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device = torch.device("xpu")
            device_count = torch.xpu.device_count()
            dtype = torch.float16 if dtype is None else dtype
        else:
            if quantize:
                raise ValueError("quantization is not available on CPU")

            device = torch.device("cpu")
            dtype = torch.float32 if dtype is None else dtype

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype,
            device_map=("auto" if device_count > 1 else None),
            load_in_8bit=quantize == "bitsandbytes",
            trust_remote_code=trust_remote_code,
        )
        if device_count == 1 and quantize != "bitsandbytes":
            model = model.to(device)

        if tokenizer.pad_token_id is None:
            if model.config.pad_token_id is not None:
                tokenizer.pad_token_id = model.config.pad_token_id
            elif model.config.eos_token_id is not None and isinstance(
                model.config.eos_token_id, int
            ):
                tokenizer.pad_token_id = model.config.eos_token_id
            elif tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self = cls.__new__(
            cls,
        )
        self.batch_class = CausalLMBatch
        super().__init__(
            self,
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            requires_padding=True,
            dtype=dtype,
            device=device,
        )
        self.quantize = quantize
        return self

    @property
    def batch_type(self) -> Type[CausalLMBatch]:
        return self.batch_class

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        token_idx,
        past_key_values: Optional[List[Tuple]] = None,
        bypass_hpu_graph: Optional[bool] = None,
    ) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]
    ]:
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "token_idx": token_idx,
        }

        # Optimum Habana got "lazy_mode" key-val only supported for llama type of models
        if self.model.config.model_type == "llama":
            kwargs["lazy_mode"] = LAZY_MODE == 1

        if self.has_position_ids:
            kwargs["position_ids"] = position_ids

        if bypass_hpu_graph is not None:
            kwargs["bypass_hpu_graphs"] = bypass_hpu_graph

        kwargs.update(self.kwargs)

        logger.info(f"input_ids: {input_ids.shape}")
        logger.info(f"attention_mask: {attention_mask.shape}")
        logger.info(f"position_ids: {position_ids.shape}")
        logger.info(f"token_idx: {token_idx}")
        # kwargs = {
        #     "input_ids": input_ids,
        #     "attention_mask": attention_mask,
        #     "past_key_values": past_key_values,
        #     "token_idx": token_idx,
        # }

        # logger.info(f"kwargs: {kwargs}")

        if past_key_values is not None and self.model.config.model_type not in [
            "gpt_bigcode"
        ]:
            return self.model.forward(**kwargs)
        else:
            outputs = self.model.forward(**kwargs)
            logger.info(f"outputs: {outputs.logits.shape}")
            return outputs.logits, outputs.past_key_values

        # outputs = self.model.forward(**kwargs)
        # if isinstance(outputs, tuple):
        #     outputs, speculative_logits = outputs
        # else:
        #     speculative_logits = None
        # return outputs.logits, speculative_logits, outputs.past_key_values

    @tracer.start_as_current_span("generate_token")
    def generate_token(
        self, batch: CausalLMBatch
    ) -> Tuple[List[Generation], Optional[CausalLMBatch], Tuple[int, int]]:
        start = time.time_ns()
        # slice the attention mask to the correct shape
        attention_mask = batch.attention_mask[:, : -batch.padding_right_offset]
        prefill = batch.past_key_values is None
        if prefill:
            # no right padding for prefill
            token_idx_scalar = attention_mask.shape[-1] - 1
            token_idx = torch.tensor(token_idx_scalar).to(self.device)
        else:
            token_idx_scalar = (
                attention_mask.shape[-1] - batch.right_padding
            )
            token_idx = torch.tensor(token_idx_scalar).to(self.device)

        logits, past = self.forward(
            batch.input_ids,
            attention_mask,
            batch.position_ids,
            token_idx,
            batch.past_key_values,
            bypass_hpu_graph=(
                prefill and self.limit_hpu_graph if self.enable_hpu_graph else None
            ),
        )

        # Results
        generations: List[Generation] = []
        stopped = True

        # Speculation is not active for causal
        accepted_ids = torch.ones_like(batch.input_ids)[:, 0]
        batch_top_token_ids, batch_top_token_logprobs = batch_top_tokens(
            batch.top_n_tokens,
            batch.top_n_tokens_tensor,
            torch.log_softmax(logits[:, -1], -1),
            accepted_ids,
        )

        start_decode = time.time_ns()

        # Zipped iterator
        iterator = zip(
            batch.requests,
            batch.input_lengths,
            batch.prefix_offsets,
            batch.read_offsets,
            logits,
            batch.next_token_choosers,
            batch.stopping_criterias,
            batch.all_input_ids,
            batch.top_n_tokens,
            batch_top_token_ids,
            batch_top_token_logprobs,
        )

        # For each member of the batch
        for i, (
            request,
            input_length,
            prefix_offset,
            read_offset,
            logits,
            next_token_chooser,
            stopping_criteria,
            all_input_ids,
            top_n_tokens,
            top_token_ids,
            top_token_logprobs,
        ) in enumerate(iterator):
            # Select next token
            next_token_id, logprobs = next_token_chooser(
                all_input_ids.view(1, -1), logits[-1:, :]
            )

            # Append next token to all tokens
            all_input_ids = torch.cat([all_input_ids, next_token_id])
            new_input_length = input_length + 1

            # Generated token
            next_token_logprob = logprobs[-1, next_token_id]
            next_token_id_squeezed = next_token_id.squeeze()
            next_token_text, prefix_offset, read_offset = self.decode_token(
                all_input_ids[:, 0], prefix_offset, read_offset
            )

            # Evaluate stopping criteria
            stop, reason = stopping_criteria(
                next_token_id_squeezed,
                next_token_text,
            )

            if not stop:
                stopped = False

            # Shard generations
            # All generations will be appended in the rust sharded client
            if i % self.world_size == self.rank:
                if stop:
                    # Decode generated tokens
                    output_text, _, _ = self.decode_token(
                        all_input_ids[:, 0],
                        prefix_offset=len(all_input_ids)
                        - stopping_criteria.current_tokens
                        - 1,
                        read_offset=len(all_input_ids)
                        - stopping_criteria.current_tokens,
                        skip_special_tokens=True,
                    )
                    # Get seed
                    if isinstance(next_token_chooser.choice, Sampling):
                        seed = next_token_chooser.choice.seed
                    else:
                        seed = None

                    generated_text = GeneratedText(
                        output_text, stopping_criteria.current_tokens, reason, seed
                    )
                else:
                    generated_text = None

                # Prefill
                if stopping_criteria.current_tokens == 1 and request.prefill_logprobs:
                    # Remove generated token to only have prefill and add nan for first prompt token
                    logprobs1 = torch.log_softmax(logits, -1)
                    logger.info(f"logprobs1: {logprobs1.shape}")
                    logger.info(f"logits={logits.shape}")
                    logger.info(f"all_input_ids={all_input_ids.shape}")
                    prefill_logprobs = [float("nan")] + torch.log_softmax(
                        logits, -1
                    ).gather(1, all_input_ids[1:]).squeeze(1)[
                        -new_input_length:-1
                    ].tolist()
                    prefill_token_ids = all_input_ids[-new_input_length:-1]
                    prefill_texts = self.tokenizer.batch_decode(
                        prefill_token_ids,
                        clean_up_tokenization_spaces=False,
                        skip_special_tokens=False,
                    )
                    prefill_tokens = Tokens(
                        prefill_token_ids,
                        prefill_logprobs,
                        prefill_texts,
                        is_special=[],
                    )
                else:
                    prefill_tokens = None

                if top_n_tokens > 0:
                    all_top_tokens = []
                    for top_token_ids, top_token_logprobs in zip(
                        top_token_ids, top_token_logprobs
                    ):
                        toptoken_texts = self.tokenizer.batch_decode(
                            top_token_ids,
                            clean_up_tokenization_spaces=False,
                            skip_special_tokens=False,
                        )
                        special_toptokens = [
                            token_id in self.all_special_ids
                            for token_id in top_token_ids
                        ]
                        top_tokens = Tokens(
                            top_token_ids,
                            top_token_logprobs,
                            toptoken_texts,
                            special_toptokens,
                        )
                        all_top_tokens.append(top_tokens)
                    top_tokens = all_top_tokens
                else:
                    top_tokens = None

                generation = Generation(
                    request.id,
                    prefill_tokens,
                    Tokens(
                        [next_token_id_squeezed],
                        [next_token_logprob],
                        [next_token_text],
                        [next_token_id_squeezed.item() in self.all_special_ids],
                    ),
                    generated_text,
                    top_tokens,
                )

                generations.append(generation)

            # Update values
            batch.next_token_choosers[i] = batch.next_token_choosers[i].advance_grammar(
                next_token_id_squeezed.item()
            )
            batch.input_ids[i, 0] = next_token_id
            batch.all_input_ids[i] = all_input_ids
            batch.input_lengths[i] = new_input_length
            batch.prefix_offsets[i] = prefix_offset
            batch.read_offsets[i] = read_offset
            batch.max_input_length = max(batch.max_input_length, new_input_length)

        # We finished all generations in the batch; there is no next batch
        if stopped:
            forward_ns = start_decode - start
            decode_ns = time.time_ns() - start_decode
            return generations, None, (forward_ns, decode_ns)

        # Slice unused values from prefill
        batch.input_ids = batch.input_ids[:, :1]

        # Update attention_mask as we added a new token to input_ids
        batch.attention_mask[:, -batch.padding_right_offset] = 1
        # Decrease right offset
        batch.padding_right_offset -= 1

        # Update position_ids
        batch.position_ids = batch.position_ids[:, -1:] + 1

        # Update past key values
        batch.past_key_values = past

        forward_ns = start_decode - start
        decode_ns = time.time_ns() - start_decode
        return generations, batch, (forward_ns, decode_ns)

    def generate_warmup_batch(self, request, seq_len, batch_size, do_sample=False):
        batch = copy.deepcopy(request.batch)
        for req in batch.requests:
            req.truncate = seq_len

        for i in range(len(batch.requests) - batch_size):
            batch.requests.pop()

        for r in batch.requests:
            r.parameters.do_sample = do_sample
        return self.batch_type.from_pb(batch, self.tokenizer, self.dtype, self.device)

    def warmup(
        self, request: generate_pb2.WarmupRequest
    ) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        assert (
            MAX_BATCH_SIZE is not None
        ), "MAX_BATCH_SIZE is not set, it should be set in the launcher"
        start = time.time()
        MAX_BATCH_TOTAL_TOKENS = MAX_BATCH_SIZE * request.max_total_tokens
        logger.info(f"MAX_BATCH_SIZE: {MAX_BATCH_SIZE}")
        logger.info(f"MAX_BATCH_TOTAL_TOKENS: {MAX_BATCH_TOTAL_TOKENS}")
        MAX_TOTAL_TOKENS = request.max_total_tokens

        batch = self.batch_type.from_pb(
            request.batch, self.tokenizer, self.dtype, self.device
        )
        max_prefill_batch_size = batch.input_ids.shape[0]
        try:
            # max prefill batch size warmup
            _, prefill_batch, _ = self.generate_token([batch])
        except Exception:
            raise RuntimeError(
                f"Not enough memory to handle {len(batch.input_ids)} prefill tokens. "
                f"You need to decrease `--max-batch-prefill-tokens`"
            )

        del prefill_batch

        # Warmup prefill batch_size
        max_input_tokens = request.max_input_tokens
        max_exp = math.ceil(math.log(max_prefill_batch_size, BATCH_SIZE_EXPONENT_BASE))
        prefill_batch_size_list = [
            BATCH_SIZE_EXPONENT_BASE**exp
            for exp in range(
                0,
                max_exp + 1,
            )
        ]
        prefill_seqlen_list = [
            seq
            for seq in range(
                PAD_SEQUENCE_TO_MULTIPLE_OF,
                max_input_tokens,
                PAD_SEQUENCE_TO_MULTIPLE_OF,
            )
        ]
        prefill_seqlen_list.append(max_input_tokens)
        prefill_batch_size_list.sort(reverse=True)
        prefill_seqlen_list.sort(reverse=True)
        try:
            for batch_size in prefill_batch_size_list:
                for seq_len in prefill_seqlen_list:
                    batch = self.generate_warmup_batch(request, seq_len - 1, batch_size)
                    _, prefill_batch, _ = self.generate_token(batch)
        except Exception:
            prefill_batch_size_list.sort()
            prefill_seqlen_list.sort()
            raise RuntimeError(
                f"Not enough memory to run following prefill batch_size."
                f"Prefill batch size list:{prefill_batch_size_list}"
                f"Prefill sequence length list:{prefill_seqlen_list}"
                f"You need to decrease `--max-batch-prefill-tokens`"
            )
        prefill_seqlen_list.sort()
        prefill_batch_size_list.sort()
        mem_stats = get_hpu_memory_stats(self.device)
        logger.info(
            f"\nFollowing prefill warmup successfully.\n"
            f"Prefill batch size list:{prefill_batch_size_list}\n"
            f"Prefill sequence length list:{prefill_seqlen_list}\n"
            f"Memory stats: {mem_stats} "
        )

        max_decode_batch_size = math.floor(MAX_BATCH_TOTAL_TOKENS / MAX_TOTAL_TOKENS)
        max_exp = math.ceil(math.log(max_decode_batch_size, BATCH_SIZE_EXPONENT_BASE))
        decode_batch_size_list = [
            BATCH_SIZE_EXPONENT_BASE**exp for exp in range(0, max_exp + 1)
        ]
        decode_batch_size_list.sort(reverse=True)

        try:
            do_samples = [True, False]
            logger.info(f"Warmup {len(do_samples)*len(decode_batch_size_list)} cases of decode batch sizes {decode_batch_size_list}.")
            step = 0
            for do_sample in do_samples:
                for batch_size in decode_batch_size_list:
                    logger.info(f"Warmup case {step}: batch_szie={batch_size}, do_sample={do_sample}")
                    step = step + 1
                    batches = []
                    iters = math.floor(batch_size / max_prefill_batch_size)
                    for i in range(iters):
                        batch = self.generate_warmup_batch(
                            request, PAD_SEQUENCE_TO_MULTIPLE_OF - 1, max_prefill_batch_size, do_sample
                        )
                        _, prefill_batch, _ = self.generate_token(batch)
                        batches.append(prefill_batch)

                    if batch_size % max_prefill_batch_size != 0:
                        batch = self.generate_warmup_batch(
                            request,
                            PAD_SEQUENCE_TO_MULTIPLE_OF - 1,
                            batch_size % max_prefill_batch_size,
                            do_sample
                        )
                        _, prefill_batch, _ = self.generate_token(batch)
                        batches.append(prefill_batch)
                    
                    batch = self.batch_class.concatenate(batches)
                    _, decode_batch, _ = self.generate_token(batch)
                    _, decode_batch, _ = self.generate_token(decode_batch)
                    del decode_batch
                    batches.clear()

        except Exception:
            raise RuntimeError(
                f"Not enough memory to warmup decode batch_sizes({decode_batch_size_list})."
                f"You need to decrease `--max-batch-total-tokens`"
            )

        decode_batch_size_list.sort()
        max_supported_total_tokens = MAX_TOTAL_TOKENS * decode_batch_size_list[-1]
        mem_stats = get_hpu_memory_stats(self.device)
        logger.info(
            f"\nFollowing decode warmup successfully.\n"
            f"Decode batch size list:{decode_batch_size_list}\n"
            f"Memory stats: {mem_stats} "
        )

        max_input_tokens = max_input_tokens
        max_total_tokens = MAX_TOTAL_TOKENS
        end = time.time()
        logger.info(f"Warmup time: {end - start}")
        return max_supported_total_tokens, max_input_tokens, max_total_tokens
