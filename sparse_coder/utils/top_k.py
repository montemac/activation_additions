"""Functions for processing autoencoders into top-k tokens."""


from collections import defaultdict

import torch as t
from accelerate import Accelerator
from transformers import AutoTokenizer


# %%
# Calculate per-input-token summed activation, for each feature dimension.
def calculate_effects(
    token_ids: list[list[int]],
    feature_activations: list[t.Tensor],
    tokenizer: AutoTokenizer,
    accelerator: Accelerator,
) -> defaultdict[int, defaultdict[str, float]]:
    """Calculate the per input token activation for each feature."""
    # The argless lambda always returns the nested defaultdict.
    feature_values = defaultdict(lambda: defaultdict(list))

    # Extract every token id into a list.
    ordered_all_ids: list[int] = [
        id for sublist in token_ids for id in sublist
    ]
    unordered_unique_ids: list[int] = list(set(ordered_all_ids))
    # Tensorize the list of ids.
    ordered_ids_tensor: t.Tensor = accelerator.prepare(
        t.tensor(ordered_all_ids)
    )

    feature_activations_parallelized: list[t.Tensor] = [
        accelerator.prepare(tensor) for tensor in feature_activations
    ]

    all_activations: t.Tensor = accelerator.prepare(
        t.cat(feature_activations_parallelized, dim=0)
    )
    # Shape (num_activations, PROJECTION_DIM).

    for i in unordered_unique_ids:
        mask: t.Tensor = ordered_ids_tensor == i
        masked_activations: t.Tensor = all_activations[mask]

        # Sum along the number of instances (dim=0).
        mean_activations = t.mean(masked_activations, dim=0)

        tkn_string = tokenizer.convert_ids_to_tokens(i)

        for dim, avg_act in enumerate(mean_activations):
            feature_values[dim][tkn_string] = avg_act.item()

    return feature_values
