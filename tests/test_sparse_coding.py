"""Unit tests for the `sparse_coding` submodule."""


from unittest.mock import patch

import numpy as np
import pytest
import torch as t
import transformers
import yaml

from sparse_coding.acts_collect import (
    shuffle_answers,
    unhot,
    pad_activations,
)

from sparse_coding.autoencoder import (
    padding_mask,
    ActivationsDataset,
    Autoencoder,
)

from sparse_coding.feature_tokens import (
    Encoder,
    unpad_activations,
    project_activations,
    select_top_k_tokens,
    round_floats,
    populate_table,
)

from sparse_coding.utils.top_k import calculate_effects
