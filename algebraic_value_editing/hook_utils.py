from typing import List, Union, Optional, Tuple

import torch
import numpy as np
from jaxtyping import Float, Int
import funcy as fn 

import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities

# Have helpers for modifying the network's activations

def get_add_activation_fn(input: str, hook_name : str, coeff : float = 1.0):
    """ Add the activations from input to the activations at the hook point. 
    
    Args:
        input: The input to add to the activations at the hook point. 
        hook_name: The name of the hook point to add the input to.
        coeff: The coefficient to multiply the input by before adding it to the activations.
    Returns:
        A hook function that adds the input to the activations at the hook point.
    """