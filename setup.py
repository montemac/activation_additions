from setuptools import setup, find_packages

setup(
    name="algebraic_value_editing",
    description=(
        "Tools for testing the algebraic value-editing conjecture (AVEC) on"
        " language models"
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        (
            "transformer-lens @"
            " git+https://github.com/montemac/TransformerLens.git@74575aeeb8cc0ac0c98a2a24014166bcde5df283"
        ),
        "torch==1.13.1",
        "numpy>=1.22.1",
        "pandas>=1.4.4",
        "jaxtyping>=0.2.14",
        "prettytable>=3.6.0",
        "funcy>=2.0",
        "wandb==0.13.5",  # transformer_lens 0.0.0 requires <0.14.0, >=0.13.5
        "openai>=0.27.2",
        "nltk>=3.8.1",
    ],
    extras_require={
        "dev": [
            "pytest",
            "notebook",  # liked by vscode
        ]
    },
)
