""" Tools for analyzing the results of algebraic value editing. """

import openai
import re
import numpy as np
import pandas as pd
from ipywidgets import widgets
from IPython.display import display
from typing import Optional, Callable

from algebraic_value_editing import completion_utils


def rate_completions(
    data_frame: pd.DataFrame,
    criterion: str = "happy",
) -> None:
    """Prompt the user to rate the generated completions, without
    indicating which condition they came from. Modifies the `data_frame`
    in place.

    args:
        `data_frame`: The `DataFrame` should have the following columns:
                `prompts`: The prompts used to generate the completions.
                `completions`: The generated completions.
                `is_modified`: Whether the completion was generated
                using a modified forward pass.

        `criterion`: The criterion to use for rating the completions.
    TODO create unit tests
    """
    # Prepare the user
    print(
        "The model was run with the bolded text as the prompt. Please rate the"
        " completions below.\n\n"
    )
    prompt: str = data_frame["prompts"].tolist()[0]
    print(f"Prompt: {completion_utils.bold_text(prompt)}\n")

    criterion_fstr: str = f"To what extent is this completion {criterion}?"
    criterion_fmt: str = completion_utils.bold_text(criterion_fstr)
    print(criterion_fmt)

    completion_box: widgets.Textarea = widgets.Textarea(
        value="Enter text here",
        layout=widgets.Layout(width="400px", height="200px"),
        disabled=True,
    )
    display(completion_box)

    # Stop the disabled text box from being grayed out
    completion_box.add_class("custom-textarea")
    custom_css = """
    <style>
        .custom-textarea textarea:disabled {
            color: black;
            opacity: 1;
        }
    </style>
    """

    display(widgets.HTML(custom_css))

    # Prompt the user to rate the completions, selected in random order
    for i in np.random.permutation(len(data_frame)):
        row: pd.Series = data_frame.iloc[i]

        # Print the criterion
        completion_box.value = row["completions"]

        # Get a rating which is an integer between 1 and 5
        rating: int = 0
        # Keep prompting until we get a valid rating
        while not 1 <= rating <= 5:
            rating = int(input("Please enter a rating (1, 2, 3, 4, 5): "))

        # Save the rating
        data_frame.loc[i, "rating"] = rating

    # Print the average ratings for each condition (modified vs. normal)
    print("Average ratings:")
    print(data_frame.groupby("is_modified")["rating"].mean())


def rate_completions_openai(
    data_frame: pd.DataFrame,
    criterion: Optional[str] = None,
    prompt_prefix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Rate completions using the OpenAI API.

    args:
        `data_frame`: The `DataFrame` should have the following columns:
                `prompts`: The prompts used to generate the completions.
                `completions`: The generated completions.
                `is_modified`: Whether the completion was generated
                using a modified forward pass.

        `criterion`: The criterion to use for rating the completions.

        `prompt_prefix`: The prefix to use for the prompt. If `None`, the
                default prompt is created from the `criterion` argument.

    returns:
        The `data_frame` with the following columns added:
            `rating`: The rating given to the completion.
            `rating_text`: The text of the rating.
    """
    for col in ("prompts", "completions", "is_modified"):
        assert col in data_frame.columns, f"Missing column: {col}"

    assert (
        criterion ^ prompt_prefix  # type:ignore
    ), "Must specify either criterion or prompt_prefix, but not both"

    # Prompt template
    # NOTE: Few-shot would work better. Might require increasing max tokens.
    prompt_prefix = (
        prompt_prefix
        or f'To what extent is this text {criterion}? Give your reasoning, followed by saying "the rating is X" for a rating X between 1 and 10. The text is:'
    )
    create_prompt: Callable = lambda text: f"{prompt_prefix}\n{text}"

    # Cost calculation for text-davinci-003 (infeasible)
    MAX_TOKENS: int = 1024
    cost_upper_bound: float = 0.002 * (MAX_TOKENS / 1024) * len(data_frame)
    print(f"Cost upper bound: ${cost_upper_bound}")

    # Send a single batched inference request
    response: pd.DataFrame = openai.Completion.create(
        model="text-curie-001",
        prompt=[
            create_prompt(row["completions"])
            for _, row in data_frame.iterrows()
        ],
        temperature=0,
    )

    # Extract the rating from message contents
    for choice in response["choices"]:
        content = choice["text"]
        match = re.search(r"[rR]ating is (\d)", content) or re.search(
            r"rated (\d)", content
        )
        rating: Optional[int] = int(match.group(1)) if match else None

        # Save the rating
        index: int = choice["index"]
        print(index, rating)
        data_frame.loc[index, "rating"] = rating
        data_frame.loc[index, "rating_text"] = content

    # Print the average ratings for each condition (modified vs. normal)
    print("Average ratings:")
    print(data_frame.groupby("is_modified")["rating"].mean())

    return data_frame
