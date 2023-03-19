import warnings
from typing import List, Union, Optional, Tuple

import torch
import torch.nn.functional
import numpy as np
import pandas as pd
from jaxtyping import Float, Int
import prettytable
from ipywidgets import Output

import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities

def hello_test():
    print('Hello!')

def get_x_vector(model, 
                 x_vector_defs : List[Tuple[Union[Tuple[str, str], str], float]], 
                 act_name,
                 pad_method='tokens_right'):
    '''Takes a list of x-vector definitions in the form ((strA, strB), coeff) and makes 
    a single summed x-vector.  Argument pad_method can be "tokens_left" or "tokens_right" which 
    will space-pad all the tokenized string inputs to be the same length, or "vectors" which will 
    zero-pad the resulting vectors; "vectors_left" and "vectors_right" apply zero-padding in 
    activation space instead.'''
    assert pad_method in ['tokens_right', 'tokens_left', 'vectors_right', 'vectors_left'], \
        'Unsupported pad method provided'
    # Convert input into single list of (str, coeff) pairs
    strs = []
    coeffs = []
    for str1or2, coeff in x_vector_defs:
        if isinstance(str1or2, tuple):
            strs.append(str1or2[0]), coeffs.append(coeff)
            strs.append(str1or2[1]), coeffs.append(-coeff)
        else:
            strs.append(str1or2), coeffs.append(coeff)
    # Get token seqs for each string
    tokens = [model.to_tokens(ss) for ss in strs]
    # Pad tokens if specified
    if pad_method.startswith('tokens'):
        SPACE_TOKEN = model.to_tokens(' ')[0, -1]
        max_token_len = max([toks.shape[-1] for toks in tokens])
        tokens = [torch.nn.functional.pad(toks, 
            (0, max_token_len-toks.shape[-1]) if pad_method=='tokens_right' else (max_token_len-toks.shape[-1], 0),
            'constant', SPACE_TOKEN) for toks in tokens]
    # Run forward passes
    # TODO: if padding at token level, can do these all in one batch, but likely not a big bottleneck right now
    caches = [model.run_with_cache(toks, names_filter=lambda ss: ss==act_name)[1] for toks in tokens]
    # Apply coeffs and calculate x-vectors
    xvectors = [coeff*cache[act_name] for coeff, cache in zip(coeffs, caches)]
    # Make final xvector, padding if needed
    if pad_method.startswith('vectors'):
        max_vector_len = max([xvec.shape[1] for xvec in xvectors])
        xvector_final = torch.zeros((xvectors[0].shape[0], max_vector_len, xvectors[0].shape[2]), 
                                    dtype=xvectors[0].dtype, device=model.cfg.device)
        for xvec in xvectors:
            if pad_method=='vectors_right':
                xvector_final[:,:xvec.shape[1],:] += xvec
            elif pad_method=='vectors_left':
                xvector_final[:,-xvec.shape[1]:,:] += xvec
    else:
        xvector_final = sum(xvectors)
    return xvector_final

    # x_vectors = []
    # for (strA, strB), coeff in x_vector_defs:
    #     # Embed to tokens
    #     a_tokens, b_tokens = [model.to_tokens(strX) for strX in (strA, strB)]

    #     # Pad to make sure token seqs are the same length
    #     if a_tokens.shape != b_tokens.shape:
    #         SPACE_TOKEN = model.to_tokens(' ')[0, -1]
    #         len_diff = a_tokens.shape[-1] - b_tokens.shape[-1]
    #         if len_diff > 0: # Add to b_tokens
    #             b_tokens = torch.tensor(b_tokens[0].tolist() + [SPACE_TOKEN] * abs(len_diff), 
    #                                     dtype=torch.int64, device=model.cfg.device).unsqueeze(0)
    #         else: 
    #             a_tokens = torch.tensor(a_tokens[0].tolist() + [SPACE_TOKEN] * abs(len_diff), 
    #                                     dtype=torch.int64, device=model.cfg.device).unsqueeze(0)
    #     assert a_tokens.shape == b_tokens.shape, f"Need same shape to compute an X-vector; instead, we have strA shape of {a_tokens.shape} and baseline shape of {b_tokens.shape}"

    #     # Run forward passes
    #     # TODO: do this in one call batching a/b together
    #     _, a_cache = model.run_with_cache(a_tokens, names_filter=lambda ss: ss==act_name)
    #     _, b_cache = model.run_with_cache(b_tokens, names_filter=lambda ss: ss==act_name)

    #     x_vectors.append(coeff*(a_cache[act_name] - b_cache[act_name]))

    # # Make and return the summed x-vector, zero-padding individual components as needed
    # for x_vector in x_vectors:

    # return sum(x_vectors)

def get_x_vector_fn(x_vector):
    '''Takes x_vector and returns a hook function that add this to the existing activations at the hook point.'''
    # def x_vector_hook(resid_pre: Float[torch.Tensor, "batch pos d_model"], hook: HookPoint) -> Float[torch.Tensor, "batch pos d_model"]:
    #     # Each HookPoint has a name attribute giving the name of the hook.
    #     x_vector = a_cache[hook.name] - b_cache[hook.name]
    #     x_vec_len = x_vector.shape[1]
        
    #     resid_pre[..., :x_vec_len, :] = coeff*x_vector + resid_pre[..., :x_vec_len, :] # Only add to first bit of the stream
    #     return resid_pre
    
    # Create and return the hook function
    def x_vector_hook_resize(resid_pre: Float[torch.Tensor, "batch pos d_model"], 
                             hook: HookPoint) -> Float[torch.Tensor, "batch pos d_model"]:
        '''Add x_vector to the output; if x_vector covers more residual streams than resid_pre (shape [batch, seq, hidden_dim]), 
        then applies only to the available residual streams.'''
        x_vec_len = x_vector.shape[1]
        if x_vec_len > resid_pre.shape[-2]: return resid_pre # NOTE If computing last new set of vectors due to caching?
        # NOTE this is going to fail when context window starts rolling over
        resid_pre[..., :x_vec_len, :] = x_vector + resid_pre[..., :x_vec_len, :] # Only add to first bit of the stream
        return resid_pre 
    
    return x_vector_hook_resize

def prompt_to_tokens(model, prompt : Union[str, List[str]]):
    if isinstance(prompt, str):
        prompt = [prompt]
    return model.to_tokens(prompt)

def complete_prompt_normal( 
        model,
        prompt : Union[str, List[str]], 
        completion_length : int = 40, 
        random_seed : Optional[int] = None,
        **sampling_kwargs):
    
    target_tokens = prompt_to_tokens(model, prompt)

    # Set seeds if provided
    if random_seed is not None:
        torch.manual_seed(random_seed)

    # Get the completions
    model.remove_all_hook_fns()
    completion = model.generate(target_tokens, max_new_tokens=completion_length, verbose=False, **sampling_kwargs)
    loss = model(completion, return_type="loss", 
            loss_per_token=True).mean(axis=1).detach().cpu().numpy()

    return [model.to_string(compl[1:]) for compl in completion], loss, target_tokens

def complete_prompt_with_x_vector(
        model, 
        prompt : Union[str, List[str]], 
        recipe: Optional[List[Tuple[Tuple[str, str], float]]] = None,
        x_vector : Optional[Float[torch.Tensor, "batch pos d_model"]] = None,
        completion_length : int = 40, 
        layer_num : int=6, 
        pad_method : str="tokens_right",
        control_type : Optional[str] = None,
        random_seed : Optional[int] = None,
        include_normal : bool = True,
        **sampling_kwargs):
    ''' Compare the model with and without hooks at layer_num, sampling completion_length additional tokens given initial prompt.
    The hooks are specified by a list of ((promptA, promptB), coefficient) tuples, which creates a net x-vector, or a 
    pre-calculated x-vector can be passed instead.  If control_type is not None, it should be a string specifying
    a type of control to use (currently only 'randn' is supported).

    Returns a tuple of completions as Tensors of tokens, with control completion included if control_type is set.
    '''
    assert recipe is None or x_vector is None, 'Only one of recipe, x_vector can be provided'
    assert recipe is not None or x_vector is not None, 'One of recipe, x_vector must be provided'

    # Get the normal completions if requested
    if include_normal:
        normal_completion_str, normal_loss, target_tokens = complete_prompt_normal(model, prompt, completion_length,
            random_seed, **sampling_kwargs)

    # Patch the model 
    act_name = utils.get_act_name("resid_pre", layer_num)
    if x_vector is None:
        x_vector = get_x_vector(model, recipe, act_name, pad_method)
    x_vector_fn = get_x_vector_fn(x_vector)
    model.add_hook(name=act_name, hook=x_vector_fn)

    # Set seeds
    if random_seed is not None:
        torch.manual_seed(random_seed)

    # Run the patched model
    patched_completion = model.generate(target_tokens, max_new_tokens=completion_length, verbose=False, **sampling_kwargs)
    patched_loss = model(patched_completion, return_type="loss",
            loss_per_token=True).mean(axis=1).detach().cpu().numpy()
    model.remove_all_hook_fns()

    # Set seeds again
    if random_seed is not None:
        torch.manual_seed(random_seed)

    # Run a control-patched model if desired
    if control_type == 'randn':
        warnings.warn('Control not supported yet')
    model.remove_all_hook_fns()

    # Put the completions into a DataFrame and return
    results = pd.DataFrame({
        'prompt': prompt,
        'patched_completion': [model.to_string(compl[1:]) for compl in patched_completion],
        'patched_loss': patched_loss
    })

    if include_normal:
        results['normal_completion'] = normal_completion_str
        results['normal_loss'] = normal_loss

    return results


def print_n_comparisons(num_comparisons : int = 5, **kwargs): # TODO batch? 
    ''' Pretty-print num_comparisons generations from patched and unpatched. Takes parameters for get_comparison. '''
    # Update the table live
    output = Output()
    display(output)

    # Generate the table
    table = prettytable.PrettyTable()
    table.align = "l"
    table.field_names = [f'\033[1mPatched completion\033[0m', f'\033[1mNormal completion\033[0m'] 

    # Ensure text has appropriate width
    width = 60
    table._min_width = {fname : width for fname in table.field_names}
    table._max_width = {fname : width for fname in table.field_names}
    # Separate completions
    table.hrules = prettytable.ALL

    # Create the repeated prompt list
    prompt = kwargs['prompt']
    del kwargs['prompt']
    prompts_list = [prompt]*num_comparisons

    # Get the completions
    results = complete_prompt_with_x_vector(prompt=prompts_list, **kwargs)

    # Formatting function
    def apply_formatting(str_):
        completion = ''.join(str_.split(prompt)[1:])
        return f'\033[1m{prompt}\033[0m{completion}'
    
    # Put into table
    for rr, row in results.iterrows():
        patch_str = apply_formatting(row['patched_completion'])
        normal_str = apply_formatting(row['normal_completion'])
        table.add_row([patch_str, normal_str])

    with output:
        output.clear_output()
        print(table)