from setuptools import setup, find_packages

setup(
    name="ricperceiver",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "opencv-python",
        "albumentations",
        "transformers",
        "einops",
        "rotary-embedding-torch",
        "wandb",
        "tqdm",
    ],
)
