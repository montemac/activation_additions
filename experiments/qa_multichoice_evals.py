# %%
"""Collect activations on Truthful-QA multiple-choice.

Requires a HuggingFace access token for the `Llama-2` models.
"""


import numpy as np
import torch as t
import transformers
from accelerate import Accelerator
from datasets import load_dataset
from numpy import ndarray
from transformers import (
    AutoModelForCausalLM,
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
MAX_NEW_TOKENS: int = 1
NUM_RETURN_SEQUENCES: int = 1
NUM_SHOT: int = 6
NUM_DATAPOINTS: int = 25  # Number of questions evaluated.
LAYER_SAMPLED: int = 16  # Layer to collect activations from.

assert (
    NUM_DATAPOINTS > NUM_SHOT
), "There must be a question not used for the multishot demonstration."

# %%
# Reproducibility.
t.manual_seed(SEED)
np.random.seed(SEED)

# %%
# Efficient inference and model parallelization.
t.set_grad_enabled(False)
accelerator: Accelerator = Accelerator()
# `device_map="auto` helps initialize big models.
model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
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
dataset: dict = load_dataset("truthful_qa", "multiple_choice")

assert (
    len(dataset["validation"]["question"]) >= NUM_DATAPOINTS
), "More datapoints sampled than exist in the dataset."

sampled_indices: ndarray = np.random.choice(
    len(dataset["validation"]["question"]),
    size=NUM_DATAPOINTS,
    replace=False,
)

sampled_indices: list = sampled_indices.tolist()

# %%
# The model answers questions on the `multiple-choice 1` task.
answers: list = []
activations: list = []

for question_num in sampled_indices:
    multishot: str = ""
    # Sample multishot questions that aren't the current question.
    multishot_indices: ndarray = np.random.choice(
        [x for x in range(len(dataset["validation"]["question"])) if x != question_num],
        size=NUM_SHOT,
        replace=False,
    )

    # Build the multishot question.
    for mult_num in multishot_indices:
        multishot += "Q: " + dataset["validation"]["question"][mult_num] + "\n"

        for choice_num in range(len(dataset["validation"]["mc1_targets"][mult_num]["choices"])):
            multishot += (
                "(" + str(choice_num) + ") "
                + dataset["validation"]["mc1_targets"][mult_num]["choices"][choice_num]
                + "\n"
            )

        labels_one_hot: list = dataset["validation"]["mc1_targets"][mult_num]["labels"]
        # Get a label int from the `labels` list.
        # Lists are 0-indexed, but I want 1-indexed options.
        correct_answer: int = np.argmax(labels_one_hot) + 1
        # Add on the correct answer under each multishot question.
        multishot += "A: (" + str(correct_answer) + ")\n"

    # Build the current question.
    question: str = "Q: " + dataset["validation"]["question"][question_num] + "\n"
    for option_num in range(
        len(dataset["validation"]["mc1_targets"][question_num]["choices"])
    ):

        question += (
            "(" + str(option_num) + ") "
            + dataset["validation"]["mc1_targets"][question_num]["choices"][option_num]
            + "\n"
        )

    question += "A: ("

    # Tokenize and prepare the model input.
    model_input: t.Tensor = tokenizer.encode(multishot + question, return_tensors="pt")
    print(multishot + question)
    model_input = accelerator.prepare(model_input)

    model_output, model_hidden = model.generate(
        model_input,
        max_new_tokens=MAX_NEW_TOKENS,
        num_return_sequences=NUM_RETURN_SEQUENCES,
        show_hidden_states=True,
    )

    answers.append(tokenizer.decode(model_output[0], skip_special_tokens=True))
    activations.append(model_hidden["hidden_states"][LAYER_SAMPLED])

    print(model_hidden["hidden_states"][LAYER_SAMPLED].shape)
    print(model_hidden["hidden_states"][LAYER_SAMPLED])

# %%
# Keep only the last line's new token in each answer.
for indx, answer in enumerate(answers):
    answers[indx] = answer.split("\n")[-1]
    answers[indx] = answers[indx].replace("A: (", "")

# %%
# Grade the model's answers.
model_accuracy: int = 0
print(*answers)

# %%
# Concat and store the model activations.
