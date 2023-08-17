# %%
"""WIP: Sort basis directions in the encoder by impact on truthful_qa score."""


from contextlib import contextmanager

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
DECODER_PATH: str = ""
SEED: int = 0
MAX_NEW_TOKENS: int = 1
NUM_RETURN_SEQUENCES: int = 1
NUM_SHOT: int = 6
NUM_DATAPOINTS: int = 20  # Number of questions evaluated.
INJECTION_LAYER: int = 16  # Layer to add activations to.

assert (
    NUM_DATAPOINTS > NUM_SHOT
), "There must be a question not used for the multishot demonstration."

# %%
# Reproducibility.
t.manual_seed(SEED)
np.random.seed(SEED)

# %%
# Efficient inference and model parallelization.
# `device_map="auto"` helps to initialize big models.
t.set_grad_enabled(False)
model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    use_auth_token=HF_ACCESS_TOKEN,
    output_hidden_states=True,
)

tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    use_auth_token=HF_ACCESS_TOKEN,
)

accelerator: Accelerator = Accelerator()
model: PreTrainedModel = accelerator.prepare(model)
model.eval()

# %%
# Load the TruthfulQA multichoice dataset.
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
# TODO: Load and preprocess the decoder.
decoder: t.Tensor = t.load(DECODER_PATH)
print(decoder.shape)
# for d in decoder.size(1):


# %%
# Convert one-hot labels to int indices.
def unhot(labels: list) -> int:
    """Change the one-hot ground truth labels to a 1-indexed int."""
    return np.argmax(labels) + 1


# %%
# TODO: Register hooks for the decoder basis dimensions.
@contextmanager
def register_feat_hooks(feature: t.Tensor, layer_num: int):
    """Project decoder basis directions into the model's activation space."""


# %%
# The model completes the `multiple-choice 1` task.
activations: list = []
answers_with_rubric: dict = {}

for question_num in sampled_indices:
    multishot: str = ""
    # Sample multishot questions that aren't the current question.
    multishot_indices: ndarray = np.random.choice(
        [
            x
            for x in range(len(dataset["validation"]["question"]))
            if x != question_num
        ],
        size=NUM_SHOT,
        replace=False,
    )

    # Build the multishot question.
    for mult_num in multishot_indices:
        multishot += "Q: " + dataset["validation"]["question"][mult_num] + "\n"

        for choice_num in range(
            len(dataset["validation"]["mc1_targets"][mult_num]["choices"])
        ):
            # choice_num is 0-indexed, but I want to display 1-indexed options.
            multishot += (
                "("
                + str(choice_num + 1)
                + ") "
                + dataset["validation"]["mc1_targets"][mult_num]["choices"][
                    choice_num
                ]
                + "\n"
            )

        labels_one_hot: list = dataset["validation"]["mc1_targets"][mult_num][
            "labels"
        ]
        # Get a label int from the `labels` list.
        correct_answer: int = unhot(labels_one_hot)
        # Add on the correct answer under each multishot question.
        multishot += "A: (" + str(correct_answer) + ")\n"

    # Build the current question.
    question: str = (
        "Q: " + dataset["validation"]["question"][question_num] + "\n"
    )
    for option_num in range(
        len(dataset["validation"]["mc1_targets"][question_num]["choices"])
    ):
        # option_num is similarly 0-indexed, but I want 1-indexed options here too.
        question += (
            "("
            + str(option_num + 1)
            + ") "
            + dataset["validation"]["mc1_targets"][question_num]["choices"][
                option_num
            ]
            + "\n"
        )
    # I only want the model to actually answer the question, with a single
    # token, so I tee it up here with the opening parentheses to a
    # multiple-choice answer integer.
    question += "A: ("

    # Tokenize and prepare the model input.
    input_ids: t.Tensor = tokenizer.encode(
        multishot + question, return_tensors="pt"
    )
    input_ids = accelerator.prepare(input_ids)
    # Generate a completion.
    outputs = model(input_ids)

    # Get the model's answer string from its logits. We want the _answer
    # stream's_ logits, so we pass `outputs.logits[:,-1,:]`. `dim=-1` here means
    # greedy sampling _over the token dimension_.
    answer_id: t.LongTensor = t.argmax(  # pylint: disable=no-member
        outputs.logits[:, -1, :], dim=-1
    )
    model_answer: str = tokenizer.decode(answer_id)
    # Cut the completion down to just its answer integer.
    model_answer = model_answer.split("\n")[-1]
    model_answer = model_answer.replace("A: (", "")

    # Get the ground truth answer.
    labels_one_hot: list = dataset["validation"]["mc1_targets"][question_num][
        "labels"
    ]
    ground_truth: int = unhot(labels_one_hot)

    # Save the model's answer besides their ground truths.
    answers_with_rubric[question_num] = [int(model_answer), ground_truth]
    # Save the model's activations.
    activations.append(outputs.hidden_states[INJECTION_LAYER])

# %%
# Grade the model's answers.
model_accuracy: float = 0.0
for (
    question_num
) in answers_with_rubric:  # pylint: disable=consider-using-dict-items
    if (
        answers_with_rubric[question_num][0]
        == answers_with_rubric[question_num][1]
    ):
        model_accuracy += 1.0

model_accuracy /= len(answers_with_rubric)
print(f"{MODEL_DIR} accuracy:{model_accuracy*100}%.")
