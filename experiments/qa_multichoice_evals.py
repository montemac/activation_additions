# %%
"""The TruthfulQA multiple-choice evals on `Llama-2` models.

Requires a HuggingFace access token for the `Llama-2` models.
"""
import numpy as np
import torch as t
import transformers
from accelerate import Accelerator
from datasets import load_dataset
from numpy import ndarray
from torch import nn
from transformers import (
    AutoModelForMultipleChoice,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


assert (
    transformers.__version__ == "4.31.0"
), "Llama-2 70B requires at least transformers v4.31.0"

# %%
# NOTE: Don't commit your HF access token!
HF_ACCESS_TOKEN: str = ""
MODEL_DIR: str = "meta-llama/Llama-2-7b-hf"
SEED: int = 0
MAX_NEW_TOKENS: int = 50
NUM_RETURN_SEQUENCES: int = 1
NUM_SHOT: int = 6

# %%
t.manual_seed(SEED)
np.random.seed(SEED)

# %%
# Efficient inference and model parallelization.
t.set_grad_enabled(False)
accelerator: Accelerator = Accelerator()
# `device_map="auto` helps initialize big models.
model: PreTrainedModel = AutoModelForMultipleChoice.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    use_auth_token=HF_ACCESS_TOKEN,
)

tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    use_auth_token=HF_ACCESS_TOKEN,
)

# The `prepare` wrapper takes over parallelization from here on.
model: PreTrainedModel = accelerator.prepare(model)
model.eval()

# %%
# Load the TruthfulQA dataset.
dataset: dict = load_dataset("domenicrosati/TruthfulQA", "multiple_choice")

# %%
# The model answers questions on the `multiple-choice 1` task.
