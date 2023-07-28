# %%
from typing import Tuple, Callable, Optional

import numpy as np
import torch as t
import transformers

assert (
    transformers.__version__ >= "4.31.0"
), "Llama-2 70B needs at least transformers 4.31.0."

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from accelerate import Accelerator
from sklearn.metrics import accuracy_score


# %%
# The TruthfulQA train-split.
dataset = load_dataset("domenicrosati/TruthfulQA")

# %%
# NOTE: Don't commit HF tokens!
ACCESS_TOKEN: str = ""
MODEL_DIR: str = "meta-llama/Llama-2-7b-chat-hf"

SEED: int = 0
t.manual_seed(SEED)
np.random.seed(SEED)

# %%
# Efficient inference and parallelization.
t.set_grad_enabled(False)
accelerator: Accelerator = Accelerator()
model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    use_auth_token=ACCESS_TOKEN,
)
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    use_auth_token=ACCESS_TOKEN,
)
model: PreTrainedModel = accelerator.prepare(model)
tokenizer: PreTrainedTokenizer = accelerator.prepare(tokenizer)
model.eval()
model.tie_weights()

# %%
# Multiple-choice task.
