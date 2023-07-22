# %%
"""
Generate HuggingFace Llama-2 prompt completions (up to both the 13B models).

Requires a Meta/HuggingFace access token. This will eventually build to steering
Llama-2, but steering isn't yet implemented.
"""
from typing import Tuple, Callable, Optional

import numpy as np
import torch as t
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import accelerate

# %% # TODO The Llama-2 70B models generate type errors, but the other Llama-2
# models work fine.
ACCESS_TOKEN = ""  # Don't accidently commit your access token!
MODEL_DIR: str = "meta-llama/Llama-2-13b-hf"
NUM_RETURN_SEQUENCES: int = 1
MAX_NEW_TOKENS: int = 50
SEED: int = 0
DO_SAMPLE: bool = True
TEMPERATURE: float = 1.0
TOP_P: float = 0.9
REP_PENALTY: float = 2.0
PROMPT: str = "I want to kill you because "

sampling_kwargs: dict = {
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "repetition_penalty": REP_PENALTY,
}

# %%
# Declare hooking types.
PreHookFn = Callable[[nn.Module, t.Tensor], Optional[t.Tensor]]
Hook = Tuple[nn.Module, PreHookFn]
Hooks = list[Hook]

# Set torch and numpy seeds.
t.manual_seed(SEED)
np.random.seed(SEED)

t.set_grad_enabled(False)
# A wrapper from accelerate does model parallelization throughout.
accelerator = accelerate.Accelerator()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    use_auth_token=ACCESS_TOKEN,
)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    use_auth_token=ACCESS_TOKEN,
)
model, tokenizer = accelerator.prepare(model, tokenizer)
model.eval()
model.tie_weights()


# %%
def tokenize(text: str) -> dict[str, t.Tensor]:
    """Tokenize prompts onto the appropriate devices."""
    tokens = tokenizer(text, return_tensors="pt")
    return accelerator.prepare(tokens)


# %%
base_tokens = model.generate(
    tokenize(PROMPT).input_ids,
    generation_config=GenerationConfig(
        **sampling_kwargs,
        do_sample=DO_SAMPLE,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=NUM_RETURN_SEQUENCES,
    ),
)
print(base_tokens)
base_strings = [tokenizer.decode(x) for x in base_tokens]
base_string = ("\n" + "." * 80 + "\n").join(
    base_strings
)  # pylint: disable=invalid-name
print(base_string)
