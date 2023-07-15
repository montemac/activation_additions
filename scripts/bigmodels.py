# %%
"""Script to run activation engineering on Vicuna 13B."""
from contextlib import contextmanager
from typing import Tuple, Callable, Optional
import torch as t
from torch import nn
import numpy as np
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig


# %%
MODEL_DIR: str = "lmsys/vicuna-7B-v1.3"
DEVICE: str = "cuda:0"
DO_SAMPLE: bool = True
TEMPERATURE: float = 1.0
TOP_P: float = 0.9
REPETITION_PENALTY: float = 2.0
MAX_NEW_TOKENS: int = 50
SEED: int = 0
COEFF: int = 5
ACT_NAME: int = 6
PROMPT: str = "I hate you because"

prompt_add, prompt_sub = "Love ", "Hate"

t.set_grad_enabled(False)
tokenizer = LlamaTokenizer.from_pretrained(MODEL_DIR)
model = LlamaForCausalLM.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model = model.half()
model.eval()


# %%
def tokenize(prompt: str) -> dict[str, t.Tensor]:
    """Tokenize a prompt into a model input."""
    passed_input = tokenizer(prompt, return_tensors="pt")
    passed_input = {k: t.to(DEVICE) for k, t in passed_input.items()}
    return passed_input


inputs = tokenize(PROMPT)
outputs = model(**inputs)

# %%
# New type declarations
PreHookFn = Callable[[nn.Module, t.Tensor], Optional[t.Tensor]]
Hook = Tuple[nn.Module, PreHookFn]
Hooks = list[Hook]


@contextmanager
def pre_hooks(hooks: Hooks):
    """Context manager to register pre-forward hooks on a model."""
    handles = []
    try:
        handles = [mod.register_forward_pre_hook(hook) for mod, hook in hooks]
        yield
    finally:
        for handle in handles:
            handle.remove()


def get_blocks(passed_model):
    """Get the blocks of a model."""
    if isinstance(passed_model, LlamaForCausalLM):
        return passed_model.model.layers
    else:
        raise ValueError(f"Unsupported model type: {type(passed_model)}")


@contextmanager
def residual_stream(passed_model: LlamaForCausalLM, layers: Optional[list[int]] = None):
    """Context manager to track residual stream activations in the model."""
    # TODO Plausibly could be replaced by 'output_hidden_states=True' in model call.

    current_stream = [None] * len(get_blocks(passed_model))

    def _make_hook(i):
        def _hook(_, current_inputs):
            current_stream[i] = current_inputs[0]

        return _hook

    hooks = [
        (layer, _make_hook(i))
        for i, layer in enumerate(get_blocks(passed_model))
        if i in layers
    ]
    with pre_hooks(hooks):
        yield current_stream


# %%
with residual_stream(model, layers=[0]) as stream:
    outputs = model(**inputs)
print(stream)

# %%
sampling_kwargs = dict(temperature=TEMPERATURE, top_p=TOP_P)
sampling_kwargs["repetition_penalty"] = REPETITION_PENALTY
# TODO: Automatic padding


# %%
def get_resid_pre(prompt: str, layer_num: int):
    """Get the residual stream activations for a prompt."""
    with residual_stream(model, layers=[layer_num]) as working_stream:
        model_outputs = model(**tokenize(prompt))
    return working_stream[layer_num]


act_add = get_resid_pre(prompt_add, ACT_NAME)
act_sub = get_resid_pre(prompt_sub, ACT_NAME)
assert act_add.shape == act_sub.shape

act_diff = act_add - act_sub

# %%
t.manual_seed(SEED)
np.random.seed(SEED)


def _hook(_, inp):
    (resid_pre,) = inp
    if resid_pre.shape[1] == 1:
        return  # caching in model.generate for new tokens
    # Only add to the first forward pass, not the later tokens.
    ppos, apos = resid_pre.shape[1], act_diff.shape[1]
    assert apos <= ppos, f"More mod tokens ({apos}) then prompt tokens ({ppos})!"
    resid_pre[:, :apos, :] += 1 * act_diff


layer = get_blocks(model)[ACT_NAME]
with pre_hooks(hooks=[(layer, _hook)]):
    outputs = model.generate(
        **tokenize([PROMPT] * 5),
        generation_config=GenerationConfig(
            **sampling_kwargs,
            do_sample=DO_SAMPLE,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=tokenizer.eos_token_id,
        ),
    )

res_strs = [tokenizer.decode(o) for o in outputs]
print(("\n\n" + "-" * 80 + "\n\n").join(res_strs))
