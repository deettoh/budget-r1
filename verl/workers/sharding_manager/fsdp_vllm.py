# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
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

import os
import logging
import torch
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy, ShardedStateDictConfig, StateDictType, FullStateDictConfig
from torch.distributed.device_mesh import DeviceMesh

from verl.third_party.vllm import LLM
from verl.third_party.vllm import parallel_state as vllm_ps
from verl import DataProto
from verl.utils.torch_functional import (broadcast_dict_tensor, allgather_dict_tensors)
from verl.utils.debug import log_gpu_memory_usage

from .base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


def _to_hf_name(peft_name):
    """Rewrite a PEFT/FSDP param name to the HF name vLLM expects."""
    name = peft_name.replace('_fsdp_wrapped_module.', '')
    name = name.replace('base_model.model.', '')
    name = name.replace('.base_layer.', '.')
    return name


class FSDPVLLMShardingManager(BaseShardingManager):

    def __init__(self,
                 module: FSDP,
                 inference_engine: LLM,
                 model_config,
                 full_params: bool = False,
                 device_mesh: DeviceMesh = None):
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.device_mesh = device_mesh

        # Full params
        self.full_params = full_params
        if full_params:
            FSDP.set_state_dict_type(self.module,
                                     state_dict_type=StateDictType.FULL_STATE_DICT,
                                     state_dict_config=FullStateDictConfig(
                                         offload_to_cpu=True,
                                         rank0_only=False))
        else:
            FSDP.set_state_dict_type(self.module,
                                     state_dict_type=StateDictType.SHARDED_STATE_DICT,
                                     state_dict_config=ShardedStateDictConfig())

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh['dp'].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

    def _lora_merged_cpu_state_dict(self):
        """Return merged base weights on CPU without mutating the actor.

        Adds each delta in a fresh tensor, not merge_adapter, which
        would desync the FSDP flat param and crash the next forward.
        """
        # map base-weight id to its lora layer
        # capture the 4-bit quant_state to dequantize before the merge
        layers = {}
        quant = {}
        for module in self.module.modules():
            if (hasattr(module, 'get_delta_weight')
                    and hasattr(module, 'get_base_layer')
                    and hasattr(module, 'active_adapters')):
                base_w = module.get_base_layer().weight
                layers[id(base_w)] = module
                qs = getattr(base_w, 'quant_state', None)
                if qs is not None:
                    quant[id(base_w)] = (base_w, qs)
        cleaned = {}
        for name, param in self.module.named_parameters():
            if 'lora_' in name:
                continue
            if id(param) in quant:
                import bitsandbytes.functional as bnb_F
                base_w, qs = quant[id(param)]
                weight = bnb_F.dequantize_4bit(
                    base_w.data, qs).to(torch.bfloat16)
            else:
                weight = param.detach()
            layer = layers.get(id(param))
            if layer is not None:
                for adapter in layer.active_adapters:
                    weight = weight + layer.get_delta_weight(adapter)
            cleaned[_to_hf_name(name)] = weight.to('cpu')
            # free each merged tensor so deltas don't pile on 24gb
            del weight
        return cleaned

    def __enter__(self):
        # merge base+delta without merge_adapter it desyncs flat param
        is_lora = hasattr(self.module, 'merge_adapter')
        if is_lora:
            log_gpu_memory_usage('Before LoRA weight sync', logger=logger)
            params = self._lora_merged_cpu_state_dict()
        else:
            log_gpu_memory_usage('Before state_dict() in sharding manager memory', logger=logger)
            params = self.module.state_dict()
            log_gpu_memory_usage('After state_dict() in sharding manager memory', logger=logger)
            # stage full-FT weights to cpu so they don't sit on the
            # 24gb card beside vllm during the sync
            params = {
                k: v.to('cpu') if hasattr(v, 'is_cuda') and v.is_cuda else v
                for k, v in params.items()
            }
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After staging state_dict to CPU in sharding manager', logger=logger)
        # Copy, not share memory
        load_format = 'hf' if self.full_params else 'dtensor'
        self.inference_engine.sync_model_weights(params, load_format=load_format)
        log_gpu_memory_usage('After sync model weights in sharding manager', logger=logger)

        del params
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After del state_dict and empty_cache in sharding manager', logger=logger)

        # TODO: offload FSDP model weights
        # self.module.cpu()
        # torch.cuda.empty_cache()
        # if torch.distributed.get_rank() == 0:
        # print(f'after model to cpu in sharding manager memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB')

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    def __exit__(self, exc_type, exc_value, traceback):
        log_gpu_memory_usage('Before vllm offload in sharding manager', logger=logger)
        self.inference_engine.offload_model_weights()
        log_gpu_memory_usage('After vllm offload in sharding manager', logger=logger)

        # self.module.to('cuda')
        # if torch.distributed.get_rank() == 0:
        #     print(f'after actor module to cuda in sharding manager memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB')

        self.module.train()

        # add empty cache after each compute
        torch.cuda.empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def preprocess_data(self, data: DataProto) -> DataProto:
        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        data.batch = allgather_dict_tensors(data.batch.contiguous(),
                                            size=vllm_ps.get_tensor_model_parallel_world_size(),
                                            group=vllm_ps.get_tensor_model_parallel_group(),
                                            dim=0)

        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        broadcast_dict_tensor(data.batch,
                              src=vllm_ps.get_tensor_model_parallel_src_rank(),
                              group=vllm_ps.get_tensor_model_parallel_group())
        dp_rank = torch.distributed.get_rank()
        dp_size = torch.distributed.get_world_size()  # not consider torch micro-dp
        tp_size = vllm_ps.get_tensor_model_parallel_world_size()
        if tp_size > 1:
            # TODO: shall we build a micro_dp group for vllm when integrating with vLLM?
            local_prompts = data.chunk(chunks=tp_size)
            data = local_prompts[dp_rank % tp_size]
        return data
