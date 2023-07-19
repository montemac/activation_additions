"""Constants and variables for loading in models including various versions of llama."""

# we may have to use specific model names to ensure this works with transformerlens
# see https://github.com/neelnanda-io/TransformerLens/blob/218ebd6f491f47f5e2f64e4c4327548b60a093eb/transformer_lens/loading_from_pretrained.py#L419 for model names
llama_model_names: list[str] = [
    "llama-7b",
    "llama-13b",
    "llama-30b",
    "llama-65b",
]
llama_relpaths = {
    "llama-7b": "7B",
    "llama-13b": "13B",
    "llama-30b": "30B",
    "llama-65b": "65B",
}
supported_models: list[str] = [
    "gpt2-small",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
]
supported_models = supported_models + llama_model_names
# specific to mesaoptimizer's remote
LLAMA_PATH = "/mnt/ssd-2/mesaoptimizer/llama/hf/"
