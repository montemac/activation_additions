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
            " git+https://github.com/neelnanda-io/TransformerLens.git@1f65f4bf35d91677deea2331561ae6a9d2e92d38"
        ),
        "torch>=1.13.0",
        "numpy>=1.22.1",
        "pandas>=1.4.4",
        "jaxtyping>=0.2.14",
        "prettytable>=3.6.0",
        "funcy>=2.0",
    ],
)
