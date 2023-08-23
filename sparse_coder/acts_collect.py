# %%
"""
Collects model activations while running Truthful-QA multiple-choice evals.

An implementation of the Truthful-QA multiple-choice task. I'm interested in
collecting residual activations during TruthfulQA to train a variational
autoencoder on, for the purpose of finding task-relevant activation directions
in the model's residual space. The script will collect those activation tensors
and save them to disk during the eval. Requires a HuggingFace access token for
the `Llama-2` models.
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
    transformers.__version__ >= "4.31.0"
), "Llama-2 70B requires at least transformers v4.31.0"

# %%
# NOTE: Don't commit your HF access token!
HF_ACCESS_TOKEN: str = ""
MODEL_DIR: str = "meta-llama/Llama-2-7b-hf"
ACTS_SAVE_PATH: str = "acts_data/activations_dataset.pt"
SEED: int = 0
MAX_NEW_TOKENS: int = 1
NUM_RETURN_SEQUENCES: int = 1
NUM_SHOT: int = 6
NUM_DATAPOINTS: int = 817  # Number of questions evaluated.
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
    output_hidden_states=True,
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
# Convert one-hot labels to int indices.
def unhot(labels: list) -> int:
    """Change the one-hot ground truth labels to a 1-indexed int."""
    return np.argmax(labels) + 1


# %%
# The model answers questions on the `multiple-choice 1` task.
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
        # option_num is similarly 0-indexed, but I want 1-indexed options here
        # too.
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
    # stream's_ logits, so we pass `outputs.logits[:,-1,:]`. `dim=-1` here
    # means greedy sampling _over the token dimension_.
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
    activations.append(outputs.hidden_states[LAYER_SAMPLED])

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


# %%
# Save the model's activations.
def pad_activations(tensor, length) -> t.Tensor:
    """Pad activation tensors to a certain stream-dim length."""
    padding_size: int = length - tensor.size(1)
    padding: t.Tensor = t.zeros(  # pylint: disable=no-member
        tensor.size(0), padding_size, tensor.size(2)
    )
    padding: t.Tensor = accelerator.prepare(padding)
    # Concat and return.
    return t.cat([tensor, padding], dim=1)  # pylint: disable=no-member


# Find the widest model activation in the stream-dimension (dim=1).
max_size: int = max(tensor.size(1) for tensor in activations)
# Pad the activations to the widest activaiton stream-dim.
padded_activations: list[t.Tensor] = [
    pad_activations(tensor, max_size) for tensor in activations
]

# Concat and store the model activations.
concat_activations: t.Tensor = t.cat(  # pylint: disable=no-member
    padded_activations,
    dim=0,
)

t.save(concat_activations, ACTS_SAVE_PATH)
