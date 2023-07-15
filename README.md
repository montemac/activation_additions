# Algebraic value editing in pretrained language models

Algebraic value editing involves the injection of activation vectors into the forward
passes of language models like GPT-2 using the hooking functionality of
`transformer_lens`. 

# Installation
After cloning the repository, run `pip install -e .` to install the
`activation_additions` package. 

There are currently a few example scripts in the `scripts/`
directory.For example, `basic_functionality.py` generates
  modified prompts (as described below).

# Methodology

## How the vectors are generated

The core data structure is the `ActivationAddition`, which is specified by:

- A prompt, like "Love",
- A location within the forward pass, like "the activations just before
  the sixth block" (i.e. `blocks.6.hook_resid_pre`), and
- A coefficient, like 2.5.

```
love_rp = ActivationAddition(prompt="Love", coeff=2.5, act_name="blocks.6.hook_resid_pre")
```

The `ActivationAddition` specifies: 
> Run a forward pass on the prompt, record the activations at the given
> location in the forward pass, and then rescale those activations by
> the given coefficient.

Then, when future forward passes reach `blocks.6.hook_resid_pre`, a hook
function adds e.g. 2.5 times the "Love" activations to the usual activations
at that location. 

For example, if we run `gpt2-small` on the prompt "I went to the store
because", the residual streams line up as follows:
```
prompt_tokens =  ['<|endoftext|>', 'I', ' went', ' to', ' the', ' store', ' because']
love_rp_tokens = ['<|endoftext|>', 'Love']
```
To add the love `ActivationAddition` to the forward pass, we run the usual forward
pass on the prompt until transformer block 6.  At this point, consider
the first two residual streams. Namely, the `'<|endoftext|>'` residual
stream and the `'I'`/`'Love'` residual stream. We add the activations in these two
residual streams.


## X-vectors are a special kind of `ActivationAddition`

A special case of this is the "X-vector." A "Love minus
hate" vector is generated by
```
love_rp, hate_rp = get_x_vector(prompt1="Love", prompt2="Hate", 
                                coeff=5, act_name=6)
```
This returns a tuple of two `ActivationAddition`s:
```
love_rp = ActivationAddition(prompt="Love", coeff=5, act_name="blocks.6.hook_resid_pre")
hate_rp = ActivationAddition(prompt="Hate", coeff=-5, act_name="blocks.6.hook_resid_pre")
```
(This is mechanistically similar to our [cheese-](https://www.lesswrong.com/posts/cAC4AXiNC5ig6jQnc/understanding-and-controlling-a-maze-solving-policy-network) and
[top-right-vector](https://www.lesswrong.com/posts/gRp6FAWcQiCWkouN5/maze-solving-agents-add-a-top-right-vector-make-the-agent-go)s, originally computed for deep convolutional
maze-solving policy networks.)

Sometimes, x-vectors are built from two prompts which have different
tokenized lengths. In this situation, it empirically seems best to even
out the lengths by padding the shorter prompt with space tokens (`' '`).
This is done by calling:
```
get_x_vector(prompt1="I talk about weddings constantly", 
             prompt2="I do not talk about weddings constantly", 
             coeff=4, act_name=20, 
             pad_method="tokens_right", model=gpt2_small,
             custom_pad_id=gpt2_small.to_single_token(' '))
```

## Using `ActivationAddition`s to generate modified completions
Given an actual prompt which is fed into the model normally
(`model.generate(prompt="Hi!")`) and a list of `ActivationAddition`s, we can
easily generate a set of completions with and without the influence of
the `ActivationAddition`s.

```
print_n_comparisons(
    prompt="I hate you because",
    model=gpt2_xl,
    tokens_to_generate=100,
    activation_additions=[love_rp, hate_rp],
    num_comparisons=15,
    seed=42,
    temperature=1, freq_penalty=1, top_p=.3
)
```

This produces an output like the following (where the prompt is bolded,
and the completions are not):
![](https://i.imgur.com/CJc4SVt.png)

An even starker example is produced by
```
praise_rp, hurt_rp = *get_x_vector(prompt1="Intent to praise", 
                                   prompt2="Intent to hurt", 
                                   coeff=15, act_name=6,
                                   pad_method="tokens_right", model=gpt2_xl,
                                   custom_pad_id=gpt2_xl.to_single_token(' '))
print_n_comparisons(
    prompt="I want to kill you because",
    model=gpt2_xl,
    tokens_to_generate=50,
    activation_additions=[praise_rp, hurt_rp],
    num_comparisons=15,
    seed=0,
    temperature=1, freq_penalty=1, top_p=.3
)
```
![](https://i.imgur.com/ewD0IKT.png)

For more examples, consult our [Google
Colab](https://colab.research.google.com/drive/183boiXfIBEdo6ch8RwOyqIZizJd6vwDl?usp=sharing).
