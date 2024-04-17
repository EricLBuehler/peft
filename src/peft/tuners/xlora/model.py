# Copyright 2023-present the HuggingFace Inc. team.
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

import json
import os
from contextlib import contextmanager
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
from safetensors.torch import save_model  # type: ignore

from peft.tuners.tuners_utils import BaseTuner

from .. import lora
from .classifier import XLoraClassifier
from .config import XLoraConfig
from .layer import XLoRAConv2dLayer, XLoRAEmbeddingLayer, XLoRALayer, XLoRALinearLayer


@staticmethod
def apply_scalings_to_x(x: torch.Tensor, scalings_layer: torch.Tensor, adapter: int) -> torch.Tensor:
    # scalings_layer = [batch_size, seq_len, n_classes]
    scalings = scalings_layer[:, :, adapter].unsqueeze(-1)
    # scalings_layer = [batch_size, seq_len, 1]
    return x * scalings


def convert_layers_to_xlora(
    base: nn.Module,  # PeftModel
    config: XLoraConfig,
) -> tuple[int, torch.device | None, list[nn.Module]]:
    """
    Returns the number of swapped layers.
    """
    total_swapped = 0
    all_layers = []

    device = None
    for module in base.modules():
        if isinstance(module, lora.Linear):
            new_layer = XLoRALinearLayer(module, config, total_swapped)
            device = module.lora_A[next(iter(module.lora_A))].weight.device
            all_layers.append(new_layer)
            total_swapped += 1
        elif isinstance(module, lora.Embedding):
            new_layer = XLoRAEmbeddingLayer(module, config, total_swapped)
            device = module.lora_A[next(iter(module.lora_embedding_A))].weight.device
            all_layers.append(new_layer)
            total_swapped += 1
        elif isinstance(module, lora.Conv2d):
            new_layer = XLoRAConv2dLayer(module, config, total_swapped)
            device = module.lora_A[next(iter(module.lora_A))].weight.device
            all_layers.append(new_layer)
            total_swapped += 1

    return (total_swapped, device, all_layers)


class XLoraModel(BaseTuner):
    """
    Creates an X-LoRA (Mixture of LoRA experts), model from a pretrained transformers model. Currently, this X-LoRA
    implementation only works with models with a transformer architecture.

    The method is described in detail in https://arxiv.org/abs/2402.07148.

    Args:
        model ([`torch.nn.Module`]): The model to be adapted.
        config ([`XLoraConfig`]): The configuration of the Lora model.
        adapter_name (`str`): The name of the adapter, does not affect the LoRA adapter names.

    Returns:
        `torch.nn.Module`: The X-LoRA model.

    Example:
        ```py
        >>> from transformers import AutoModelForCausalLM, AutoConfig
        >>> from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_int8_training

        >>> model_config = AutoConfig.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        >>> config = XLoraConfig(
        ...     task_type="CAUSAL_LM",
        ...     hidden_size=model_config.hidden_size,
        ...     xlora_depth=4,
        ...     adapters={
        ...         "adapter_1": "./path/to/the/checkpoint/",
        ...         "adapter_2": "./path/to/the/checkpoint/",
        ...         "adapter_n": "./path/to/the/checkpoint/",
        ...     },
        ... )

        >>> model = AutoModelForCausalLM.from_pretrained(
        ...     "mistralai/Mistral-7B-Instruct-v0.1",
        ...     trust_remote_code=True,
        ...     use_flash_attention_2=False,
        ...     device_map="cuda:0",
        ...     torch_dtype=torch.bfloat16,
        ... )
        >>> model = prepare_model_for_int8_training(model)
        >>> xlora_model = get_peft_model(model, config)
        ```
    """

    def __init__(
        self,
        model: nn.Module,
        config: Union[dict[str, XLoraConfig], XLoraConfig],
        adapter_name: str,
    ) -> None:
        # a bit hacky but eh
        from peft import PeftModel

        # TODO: probably needs some more work, as some stuff is hard-coded here, subfolders not considered
        lora_model = PeftModel.from_pretrained(model, config["default"].adapters["0"], adapter_name="0")
        for key, val in config["default"].adapters.items():
            if key == "0":
                continue
            lora_model.load_adapter(val, key)

        # remove the PeftModel wrapper
        lora_model = lora_model.base_model
        # don't call super().__init__
        nn.Module.__init__(self)
        self.model = lora_model

        if isinstance(config, dict):
            conf = config[adapter_name]
        else:
            conf = config
        self.config = conf
        self.peft_config = {adapter_name: conf}

        if hasattr(model.config, "use_cache") and model.config.use_cache:
            raise ValueError("`use_cache` must be False")

        total_swapped, device, all_layers = convert_layers_to_xlora(self.model, self.config)
        n_classes = len(self.config.adapters)
        xlora_classifier = XLoraClassifier(self.model, self.config, n_classes, total_swapped, device)

        # Setup the model internal state
        self.internal_xlora_classifier = xlora_classifier

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        # Handle case during init
        if not hasattr(self, "model"):
            return
        active_adapters = []
        copy = self.model.active_adapters.copy()
        for name in self.model.active_adapters:
            if not isinstance(self.model.peft_config[name], XLoraConfig):
                active_adapters.append(name)
        self.model.active_adapter = active_adapters
        if self.config.use_trainable_adapters:
            super()._mark_only_adapters_as_trainable(model)

        self.model.active_adapter = copy

    @staticmethod
    def _prepare_adapter_config(peft_config, _model_config):
        # Handle X-LoRA case
        return peft_config

    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        # Does nothing because XLoraModel has no target modules
        pass

    @staticmethod
    def _check_target_module_exists(lora_config, key):
        # Does nothing because XLoraModel has no target modules
        return False

    def _save_pretrained_hook(
        self,
        save_directory: str,
        safe_serialization: bool = True,
        is_main_process: bool = True,
        **kwargs: Any,
    ) -> None:
        conf = self.config.__dict__.copy()

        # So that the adapters are unloadable and the user is forced to set them for from_pretrained
        conf["adapters"] = None
        if hasattr(conf, "_subfolders"):
            del conf["_subfolders"]  # It may have been added in from_pretrained
        with open(os.path.join(save_directory, "xlora_config.json"), "w") as f:
            json.dump(conf, f)

        if safe_serialization:
            # https://github.com/huggingface/peft/blob/main/src/peft/peft_model.py#L223
            if is_main_process and safe_serialization:
                save_model(
                    self.internal_xlora_classifier, os.path.join(save_directory, "xlora_classifier.safetensors")
                )
        elif is_main_process:
            state_dict = self.internal_xlora_classifier.state_dict()
            torch.save(state_dict, os.path.join(save_directory, "xlora_classifier.pt"))

    # def forward(self, *args, **kwargs):
    #     return self.lora_model.model(*args, **kwargs)

    @contextmanager
    def _enable_peft_forward_hooks(self, *args, input_ids=None, inputs_embeds=None, **kwargs):
        # here we need some more precautions to ensure that the LoRA adapters are disabled
        base_model_outputs = self.model(
            input_ids=input_ids, inputs_embeds=inputs_embeds, output_hidden_states=True, return_dict=True
        )
        xlora_scalings = self.internal_xlora_classifier(
            base_model_outputs, input_ids=input_ids, inputs_embeds=inputs_embeds
        )
        self._internal_xlora_scalings = xlora_scalings

        def hook(module, *args, **kwargs):
            kwargs["xlora_scalings"] = xlora_scalings
            return args, kwargs

        handles = []

        for module in self.model.modules():
            if isinstance(module, XLoRALayer):
                handle = module.register_forward_pre_hook(hook, with_kwargs=True)
                handles.append(handle)

        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def set_topk_lora(self, value: Optional[int]):
        """
        Sparsely select the specified top_k LoRA experts instead of the default dense method. Set to None to use dense.
        This is reflected in the config.
        """
        classifier: XLoraClassifier = self.internal_xlora_classifier  # type: ignore
        classifier.config.top_k_lora = value

    def set_global_scaling_weight(self, weight: float):
        """
        Set the global LoRA weight, a scalar to multiply the output of each LoRA adapter by. This is by default 1. This
        is reflected in the config.
        """
        classifier: XLoraClassifier = self.internal_xlora_classifier  # type: ignore
        classifier.config.global_scaling_weight = weight

    def get_global_scaling_weight(self) -> float:
        """
        Get the global LoRA weight.
        """
        classifier: XLoraClassifier = self.internal_xlora_classifier  # type: ignore
        return classifier.config.global_scaling_weight

    def get_latest_scalings(self) -> Optional[torch.Tensor]:
        """
        Returns the latest scalings prediction, or None if no scalings have been predicted. The tensor is of shape
        (batch_size, seq_len, n_layers, n_classes).
        """
        return self._internal_xlora_scalings

    def get_scalings_log(self) -> List[torch.Tensor]:
        """
        Returns a shallow (only copying the list itself not the tensors) copy of the list containing the scalings log.
        Editing the list does not change the underlying log. The tensors are of shape (batch_size, seq_len, n_layers,
        n_classes). The seq_len dim may vary with input dimension.
        """
        classifier: XLoraClassifier = self.internal_xlora_classifier  # type: ignore
        return classifier.log_scalings.copy()

    def enable_scalings_logging(self):
        """
        Enable scalings logging.
        """
        classifier: XLoraClassifier = self.internal_xlora_classifier  # type: ignore
        classifier.scalings_logging = True

    def disable_scalings_logging(self):
        """
        Disable scalings logging, without clearing the log.
        """
        classifier: XLoraClassifier = self.internal_xlora_classifier  # type: ignore
        classifier.scalings_logging = False

    def clear_scalings_log(self):
        """
        Clear the scalings log.
        """
        classifier: XLoraClassifier = self.internal_xlora_classifier  # type: ignore
        classifier.log_scalings.clear()

    def get_bucketed_scalings_log(self) -> dict[int, tuple[list[int], list[torch.Tensor]]]:
        """
        Returns bucketed scalings, bucketed by seq_len. Each value consists of the positions (the first) and the
        associated tensors. The positions are paired with the associated tensors and give the position in the scaling
        log.
        """
        classifier: XLoraClassifier = self.internal_xlora_classifier  # type: ignore
        return classifier._get_bucketed_scalings()

    def set_use_trainable_adapters(self, use_trainable_adapters: bool):
        """
        Set the adapters to trainable or not trainable.

        This is reflected in the config.
        """
        for name, param in self.named_parameters():
            if "lora_" in name:
                param.requires_grad = use_trainable_adapters

        self.config.use_trainable_adapters = use_trainable_adapters

    def get_use_trainable_adapters(self) -> bool:
        """
        Get the trainable or not trainable state of the adapters.
        """
        return self.config.use_trainable_adapters
