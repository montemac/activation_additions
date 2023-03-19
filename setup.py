from setuptools import setup, find_packages

setup(
    name='avec-gpt2',
    description='Tools for testing the algebraic value-editing conjecture (AVEC) on GPT-2 models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    version='0.2.0',
    packages=find_packages(),
    install_requires=[
        'transformer-lens @ git+https://github.com/neelnanda-io/TransformerLens.git@1f65f4bf35d91677deea2331561ae6a9d2e92d38',
        'torch>=1.13.1',
        'numpy>=1.22.1',
        'pandas>=1.4.4',
        'jaxtyping>=0.2.14',
        'prettytable>=3.6.0',
        'ipywidgets>=7.7',
    ]
)
