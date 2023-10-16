# %%
"""
Collects model activations while running Truthful-QA multiple-choice evals.

An implementation of the Truthful-QA multiple-choice task. I'm interested in
collecting residual activations during TruthfulQA to train a variational
autoencoder on, for the purpose of finding task-relevant activation directions
in the model's residual space. The script will collect those activation tensors
and their prompts and save them to disk during the eval. Requires a HuggingFace
access token for the `Llama-2` models.
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

from sparse_coding.utils.configure import load_yaml_constants


assert (
    transformers.__version__ >= "4.31.0"
), "Llama-2 70B requires at least transformers v4.31.0"

# %%
# Set up constants.
access, config = load_yaml_constants()

HF_ACCESS_TOKEN = access.get("HF_ACCESS_TOKEN", "")
MODEL_DIR = config.get("MODEL_DIR")
LARGE_MODEL_MODE = config.get("LARGE_MODEL_MODE")
PROMPT_IDS_PATH = config.get("PROMPT_IDS_PATH")
ACTS_SAVE_PATH = config.get("ACTS_DATA_PATH")
ACTS_LAYER = config.get("ACTS_LAYER")
SEED = config.get("SEED")
MAX_NEW_TOKENS = config.get("MAX_NEW_TOKENS", 1)
NUM_RETURN_SEQUENCES = config.get("NUM_RETURN_SEQUENCES", 1)
NUM_SHOT = config.get("NUM_SHOT", 6)
NUM_QUESTIONS_EVALED = config.get("NUM_QUESTIONS_EVALED", 817)

assert isinstance(LARGE_MODEL_MODE, bool), "LARGE_MODEL_MODE must be a bool."
assert (
    NUM_QUESTIONS_EVALED > NUM_SHOT
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
    token=HF_ACCESS_TOKEN,
    output_hidden_states=True,
)

tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    token=HF_ACCESS_TOKEN,
)
model: PreTrainedModel = accelerator.prepare(model)
model.eval()

# %%
# Load the TruthfulQA dataset.
dataset: dict = load_dataset("truthful_qa", "multiple_choice")

assert (
    len(dataset["validation"]["question"]) >= NUM_QUESTIONS_EVALED
), "More datapoints sampled than exist in the dataset."

sampled_indices: ndarray = np.random.choice(
    len(dataset["validation"]["question"]),
    size=NUM_QUESTIONS_EVALED,
    replace=False,
)

sampled_indices: list = sampled_indices.tolist()


# %%
# Shuffle the correct answers.
def shuffle_answers(choices, labels_one_hot):
    """Shuffle the answers and the answer labels correspondingly."""
    paired_choices = list(zip(choices, labels_one_hot))
    np.random.shuffle(paired_choices)
    choices, labels_one_hot = zip(*paired_choices)
    return choices, labels_one_hot


# %%
# Convert one-hot labels to int indices.
def unhot(labels) -> int:
    """Change the one-hot ground truth labels to a 1-indexed int."""
    return np.argmax(labels) + 1


# %%
# The model answers questions on the `multiple-choice 1` task.
activations: list = []
answers_with_rubric: dict = {}
prompts_ids: list = []

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

        # Shuffle the answers and labels.
        unshuffled_choices: list = dataset["validation"]["mc1_targets"][
            mult_num
        ]["choices"]
        unshuffled_labels: list = dataset["validation"]["mc1_targets"][
            mult_num
        ]["labels"]

        shuffled_choices, shuffled_labels = shuffle_answers(
            unshuffled_choices, unshuffled_labels
        )

        for choice_num, shuffled_choice in enumerate(shuffled_choices):
            # choice_num is 0-indexed, but I want to display 1-indexed options.
            multishot += (
                "(" + str(choice_num + 1) + ") " + shuffled_choice + "\n"
            )

        # Get a label int from the `labels` list.
        correct_answer: int = unhot(shuffled_labels)
        # Add on the correct answer under each multishot question.
        multishot += "A: (" + str(correct_answer) + ")\n"

    # Build the current question with shuffled choices.
    question: str = (
        "Q: " + dataset["validation"]["question"][question_num] + "\n"
    )

    unshuffled_choices_current: list = dataset["validation"]["mc1_targets"][
        question_num
    ]["choices"]
    unshuffled_labels_current: list = dataset["validation"]["mc1_targets"][
        question_num
    ]["labels"]

    shuffled_choices_current, shuffled_labels_current = shuffle_answers(
        unshuffled_choices_current, unshuffled_labels_current
    )

    for option_num, shuffled_option in enumerate(shuffled_choices_current):
        # option_num is similarly 0-indexed, but I want 1-indexed options here
        # too.
        question += "(" + str(option_num + 1) + ") " + shuffled_option + "\n"
    # I only want the model to actually answer the question, with a single
    # token, so I tee it up here with the opening parentheses to a
    # multiple-choice answer integer.
    question += "A: ("

    # Tokenize and prepare the model input.
    input_ids: t.Tensor = tokenizer.encode(
        multishot + question, return_tensors="pt"
    )
    prompts_ids.append(input_ids)

    # (The `accelerate` parallelization doesn't degrade gracefully with small
    # models.)
    if not LARGE_MODEL_MODE:
        input_ids = input_ids.to(model.device)

    input_ids = accelerator.prepare(input_ids)
    # Generate a completion.
    outputs = model(input_ids)

    # Get the model's answer string from its logits. We want the _answer
    # stream's_ logits, so we pass `outputs.logits[:,-1,:]`. `dim=-1` here
    # means greedy sampling _over the token dimension_.
    answer_id: t.LongTensor = t.argmax(outputs.logits[:, -1, :], dim=-1)
    model_answer: str = tokenizer.decode(answer_id)

    # Cut the completion down to just its answer integer.
    model_answer = model_answer.split("\n")[-1]
    model_answer = model_answer.replace("A: (", "")

    # Get the ground truth answer.
    ground_truth: int = unhot(shuffled_labels_current)
    # Save the model's answer besides their ground truths.
    answers_with_rubric[question_num] = [int(model_answer), ground_truth]
    # Save the model's activations.
    activations.append(outputs.hidden_states[ACTS_LAYER])

# %%
# Grade the model's answers.
model_accuracy: float = 0.0
for (
    question_idx
) in answers_with_rubric:  # pylint: disable=consider-using-dict-items
    if (
        answers_with_rubric[question_idx][0]
        == answers_with_rubric[question_idx][1]
    ):
        model_accuracy += 1.0

model_accuracy /= len(answers_with_rubric)
print(f"{MODEL_DIR} accuracy:{model_accuracy*100}%.")


# %%
# Save the model's prompt_ids and activations.
def pad_activations(tensor, length) -> t.Tensor:
    """Pad activation tensors to a certain stream-dim length."""
    padding_size: int = length - tensor.size(1)
    padding: t.Tensor = t.zeros(tensor.size(0), padding_size, tensor.size(2))

    if not LARGE_MODEL_MODE:
        padding: t.Tensor = padding.to(tensor.device)

    padding: t.Tensor = accelerator.prepare(padding)
    # Concat and return.
    return t.cat([tensor, padding], dim=1)


# Find the widest model activation in the stream-dimension (dim=1).
max_size: int = max(tensor.size(1) for tensor in activations)
# Pad the activations to the widest activaiton stream-dim.
padded_activations: list[t.Tensor] = [
    pad_activations(tensor, max_size) for tensor in activations
]

# Concat the model activations.
concat_activations: t.Tensor = t.cat(
    padded_activations,
    dim=0,
)

# Prep to save the prompt_ids.
prompt_ids_list: list = []
for question_ids in prompts_ids:
    prompt_ids_list.append(question_ids.tolist())

prompt_ids_array: ndarray = np.array(prompt_ids_list, dtype=object)

# Save the activations and prompt_ids.
np.save(PROMPT_IDS_PATH, prompt_ids_array, allow_pickle=True)
t.save(concat_activations, ACTS_SAVE_PATH)
