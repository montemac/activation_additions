"""Functions for processing autoencoders into top-k tokens."""


import textwrap
from collections import defaultdict
from math import ceil

import torch as t
from accelerate import Accelerator
from transformers import AutoTokenizer


# `per_input_token_effects` is a linchpin interpretability function. I break up
# its functionality into several tacit dependency functions in this module so
# that it's readable.
def per_input_token_effects(
    token_ids_by_q: list[list[int]],
    encoder_activations_by_q: list[t.Tensor],
    encoder,
    tokenizer: AutoTokenizer,
    accelerator: Accelerator,
    dims_per_batch: int,
    large_model_mode: bool,
) -> defaultdict[int, defaultdict[str, float]]:
    """Return the autoencoder's summed activations, at each feature dimension,
    at each input token."""

    # Begin pre-processing. Calulate the number of dimensional batches to run.
    print("Starting pre-processing...")
    num_dim_batches: int = batching_setup(dims_per_batch, encoder)

    # Initialize the effects dictionary.
    effect_scalar_by_dim_by_input_token = defaultdict(defaultdict_factory)

    # Pre-process `token_ids_by_q`.
    flat_input_token_ids, unique_input_token_ids = pre_process_input_token_ids(
        token_ids_by_q, encoder, accelerator, large_model_mode
    )

    print("Pre-processing complete!")
    effect_scalar_by_dim_by_input_token = batches_loop(
        num_dim_batches,
        dims_per_batch,
        encoder_activations_by_q,
        encoder,
        accelerator,
        tokenizer,
        effect_scalar_by_dim_by_input_token,
        unique_input_token_ids,
        flat_input_token_ids,
        large_model_mode,
    )

    return effect_scalar_by_dim_by_input_token


# Helper functions for `per_token_effects`.
def modal_tensor_acceleration(
    tensor: t.Tensor, encoder, accelerator: Accelerator, large_model_mode: bool
) -> t.Tensor:
    """Accelerate a tensor; manually move it where the accelerator fails."""
    if large_model_mode is False:
        tensor = tensor.to(encoder.encoder_layer.weight.device)
    tensor = accelerator.prepare(tensor)

    return tensor


def batching_setup(dims_per_batch: int, encoder) -> int:
    """Determine the number of dimensional batches to be run."""
    num_dim_batches: int = ceil(
        encoder.encoder_layer.weight.shape[0] / dims_per_batch
    )
    print(f"Total number of batches to be run: {num_dim_batches}")

    return num_dim_batches


def defaultdict_factory():
    """Factory for string defaultdicts."""
    return defaultdict(str)


def pre_process_input_token_ids(
    token_ids_by_q, encoder, accelerator, large_model_mode
):
    """Pre-process the `token_ids_by_q`."""

    # Flatten the input token ids.
    flat_input_token_ids = [
        input_token_id
        for question in token_ids_by_q
        for input_token_id in question
    ]

    # Deduplicate the `flat_input_token_ids`.
    unique_input_token_ids = list(set(flat_input_token_ids))

    # Tensorize and accelerate `flat_input_token_ids`.
    flat_input_token_ids = t.tensor(flat_input_token_ids)
    flat_input_token_ids = modal_tensor_acceleration(
        flat_input_token_ids, encoder, accelerator, large_model_mode
    )

    return flat_input_token_ids, unique_input_token_ids


def batches_loop(
    num_dim_batches: int,
    dims_per_batch: int,
    encoder_activations_by_q,
    encoder,
    accelerator: Accelerator,
    tokenizer: AutoTokenizer,
    effect_scalar_by_dim_by_input_token,
    unique_input_token_ids,
    flat_input_token_ids,
    large_model_mode: bool,
) -> defaultdict[int, defaultdict[str, float]]:
    """Loop over the batches while printing current progress."""

    starting_dim_index, ending_dim_index = 0, 0

    for batch in range(num_dim_batches):
        print(f"Starting batch {batch+1} of {num_dim_batches}...")

        ending_dim_index += dims_per_batch
        if ending_dim_index > encoder.encoder_layer.weight.shape[0]:
            ending_dim_index = encoder.encoder_layer.weight.shape[0]

        if batch + 1 > num_dim_batches:
            assert starting_dim_index - ending_dim_index == dims_per_batch
        elif batch + 1 == num_dim_batches:
            assert starting_dim_index - ending_dim_index <= dims_per_batch

        # Note that `batched_dims_from_encoder_activations` has
        # lost the question data that `encoder_activations_by_q` had.
        batched_dims_from_encoder_activations = (
            pre_process_encoder_activations_by_batch(
                encoder_activations_by_q,
                dims_per_batch,
                encoder,
                accelerator,
                starting_dim_index,
                ending_dim_index,
                large_model_mode,
            )
        )

        assert not t.isnan(batched_dims_from_encoder_activations).any()

        for input_token_id in unique_input_token_ids:
            input_token_string = tokenizer.convert_ids_to_tokens(
                input_token_id
            )
            dims_from_encoder_activations_at_input_token_in_batch = (
                filter_encoder_activations_by_input_token(
                    flat_input_token_ids,
                    input_token_id,
                    batched_dims_from_encoder_activations,
                )
            )
            averaged_dim_from_encoder_activations_at_input_token_in_batch = (
                average_encoder_activations_at_input_token(
                    dims_from_encoder_activations_at_input_token_in_batch,
                )
            )

            # Add the averaged activations on to the effects dictionary.
            for dim_in_batch, averaged_activation_per_dim in enumerate(
                averaged_dim_from_encoder_activations_at_input_token_in_batch
            ):
                effect_scalar_by_dim_by_input_token[
                    starting_dim_index + dim_in_batch
                ][input_token_string] = averaged_activation_per_dim.item()

        print(
            textwrap.dedent(
                f"""
                Batch {batch+1} complete: data for encoder dims indices
                {starting_dim_index} through {ending_dim_index-1} appended!
                """
            )
        )

        # Update `starting_dim_index` for the next batch.
        starting_dim_index = ending_dim_index

    return effect_scalar_by_dim_by_input_token


def pre_process_encoder_activations_by_batch(
    encoder_activations_by_q,
    dims_per_batch,
    encoder,
    accelerator,
    starting_dim_index,
    ending_dim_index,
    large_model_mode,
) -> t.Tensor:
    """Pre-process the `encoder_activations_by_q` for each batch."""
    batched_dims_from_encoder_activations: list = []

    for question_block in encoder_activations_by_q:
        batched_dims_from_encoder_activations.append(
            question_block[:, starting_dim_index:ending_dim_index]
        )

    batched_dims_from_encoder_activations = accelerator.prepare(
        batched_dims_from_encoder_activations
    )
    # Remove the question data.
    batched_dims_from_encoder_activations: t.Tensor = t.cat(
        batched_dims_from_encoder_activations, dim=0
    )

    assert batched_dims_from_encoder_activations.shape[1] <= dims_per_batch

    # Accelerate `batched_dims_from_encoder_activations`.
    batched_dims_from_encoder_activations = modal_tensor_acceleration(
        batched_dims_from_encoder_activations,
        encoder,
        accelerator,
        large_model_mode,
    )

    return batched_dims_from_encoder_activations


# Remember that dimensional batch slicing is already done coming in.
def filter_encoder_activations_by_input_token(
    flat_input_token_ids: t.Tensor,
    input_token_id: int,
    batched_dims_from_encoder_activations: t.Tensor,
):
    """Isolate just the activations at an input token id."""
    indices_of_encoder_activations_at_input_token = t.nonzero(
        flat_input_token_ids == input_token_id
    )
    flat_indices_of_encoder_activations_at_input_token = (
        indices_of_encoder_activations_at_input_token.squeeze(dim=1)
    )

    # Fancy index along dim=0.
    dims_from_encoder_activations_at_input_token_in_batch = (
        batched_dims_from_encoder_activations[
            flat_indices_of_encoder_activations_at_input_token
        ]
    )

    return dims_from_encoder_activations_at_input_token_in_batch


def average_encoder_activations_at_input_token(
    dims_from_encoder_activations_at_input_token_in_batch,
):
    """Average over encoder activations at a common input token."""

    # Average across dimensional instances.
    averaged_dim_from_encoder_activations_at_input_token_in_batch = t.mean(
        dims_from_encoder_activations_at_input_token_in_batch, dim=0
    )

    assert (
        len(
            averaged_dim_from_encoder_activations_at_input_token_in_batch.shape
        )
        == 1
    ), "Tensor has more than one dimension! It should be a vector."

    assert not t.isnan(
        averaged_dim_from_encoder_activations_at_input_token_in_batch
    ).any(), "Processed tensor contains NaNs!"

    return averaged_dim_from_encoder_activations_at_input_token_in_batch


# All other `top-k` functions below.
def project_activations(
    acts_list: list[t.Tensor],
    projector,
    accelerator: Accelerator,
) -> list[t.Tensor]:
    """Projects the activations block over to the sparse latent space."""

    # Remember the original question lengths.
    lengths: list[int] = [len(question) for question in acts_list]

    flat_acts: t.Tensor = t.cat(acts_list, dim=0)
    flat_acts: t.Tensor = accelerator.prepare(flat_acts)
    projected_flat_acts: t.Tensor = projector(flat_acts).detach()

    # Reconstruct the original question lengths.
    projected_activations: list[t.Tensor] = []
    current_idx: int = 0
    for length in lengths:
        projected_activations.append(
            projected_flat_acts[current_idx : current_idx + length, :]
        )
        current_idx += length

    return projected_activations


def select_top_k_tokens(
    effects_dict: defaultdict[int, defaultdict[str, float]],
    top_k: int,
) -> defaultdict[int, list[tuple[str, float]]]:
    """Select the top-k tokens for each feature."""
    tp_k_tokens = defaultdict(list)

    for feature_dim, tokens_dict in effects_dict.items():
        # Sort tokens by their dimension activations.
        sorted_effects: list[tuple[str, float]] = sorted(
            tokens_dict.items(), key=lambda x: x[1], reverse=True
        )
        # Add the top-k tokens.
        tp_k_tokens[feature_dim] = sorted_effects[:top_k]

    return tp_k_tokens


def unpad_activations(
    activations_block: t.Tensor, unpadded_prompts: list[list[int]]
) -> list[t.Tensor]:
    """
    Unpads activations to the lengths specified by the original prompts.

    Note that the activation block must come in with dimensions (batch x stream
    x embedding_dim), and the unpadded prompts as an array of lists of
    elements.
    """
    unpadded_activations: list = []

    for k, unpadded_prompt in enumerate(unpadded_prompts):
        try:
            original_length: int = len(unpadded_prompt)
            # From here on out, activations are unpadded, and so must be
            # packaged as a _list of tensors_ instead of as just a tensor
            # block.
            unpadded_activations.append(
                activations_block[k, :original_length, :]
            )
        except IndexError:
            print(f"IndexError at {k}")
            # This should only occur when the data collection was interrupted.
            # In that case, we just break when the data runs short.
            break

    return unpadded_activations
