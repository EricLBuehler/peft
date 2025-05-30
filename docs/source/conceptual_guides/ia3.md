<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# IA3 

This conceptual guide gives a brief overview of [IA3](https://huggingface.co/papers/2205.05638), a parameter-efficient fine tuning technique that is 
intended to improve over [LoRA](./lora).

To make fine-tuning more efficient, IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations) 
rescales inner activations with learned vectors. These learned vectors are injected in the attention and feedforward modules 
in a typical transformer-based architecture. These learned vectors are the only trainable parameters during fine-tuning, and thus the original 
weights remain frozen. Dealing with learned vectors (as opposed to learned low-rank updates to a weight matrix like LoRA)
keeps the number of trainable parameters much smaller. 

Being similar to LoRA, IA3 carries many of the same advantages: 

* IA3 makes fine-tuning more efficient by drastically reducing the number of trainable parameters. (For T0, an IA3 model only has about 0.01% trainable parameters, while even LoRA has > 0.1%)
* The original pre-trained weights are kept frozen, which means you can have multiple lightweight and portable IA3 models for various downstream tasks built on top of them.
* Performance of models fine-tuned using IA3 is comparable to the performance of fully fine-tuned models.
* IA3 does not add any inference latency because adapter weights can be merged with the base model.

In principle, IA3 can be applied to any subset of weight matrices in a neural network to reduce the number of trainable
parameters. Following the authors' implementation, IA3 weights are added to the key, value and feedforward layers
of a Transformer model. To be specific, for transformer models, IA3 weights are added to the outputs of key and value layers, and to the input of the second feedforward layer
in each transformer block.

Given the target layers for injecting IA3 parameters, the number of trainable parameters
can be determined based on the size of the weight matrices.


## Common IA3 parameters in PEFT

As with other methods supported by PEFT, to fine-tune a model using IA3, you need to:

1. Instantiate a base model.
2. Create a configuration (`IA3Config`) where you define IA3-specific parameters.
3. Wrap the base model with `get_peft_model()` to get a trainable `PeftModel`.
4. Train the `PeftModel` as you normally would train the base model.

`IA3Config` allows you to control how IA3 is applied to the base model through the following parameters:

- `target_modules`: The modules (for example, attention blocks) to apply the IA3 vectors.
- `feedforward_modules`: The list of modules to be treated as feedforward layers in `target_modules`. While learned vectors are multiplied with
the output activation for attention blocks, the vectors are multiplied with the input for classic feedforward layers. Note that `feedforward_modules` must be a subset of `target_modules`.
- `modules_to_save`: List of modules apart from IA3 layers to be set as trainable and saved in the final checkpoint. These typically include model's custom head that is randomly initialized for the fine-tuning task.

## Example Usage

For the task of sequence classification, one can initialize the IA3 config for a Llama model as follows:

```py
peft_config = IA3Config(
    task_type=TaskType.SEQ_CLS, target_modules=["k_proj", "v_proj", "down_proj"], feedforward_modules=["down_proj"]
)
```