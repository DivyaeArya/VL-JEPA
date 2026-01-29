from setuptools import find_packages, setup

requirements = [
    "torch",
    "transformers",
    "datasets",
    "numpy",
    "Pillow",
    "safetensors",
]

extras_require = {
    # 'cuda': ['torch'], # PyTorch usually handles cuda selection via index-url or specific builds
}

setup(
    name='vljepa',
    url='https://github.com/DivyaeArya/VL-JEPA',
    packages=find_packages(),
    version='0.0.1a0',
    readme="README.md",
    author_email="divyae.arya@gmail.com",
    description="VL-JEPA PyTorch (https://arxiv.org/abs/2512.10942)",
    long_description=open("README.md", encoding="utf-8").read() if open("README.md", encoding="utf-8") else "",
    long_description_content_type="text/markdown",
    author="Divyae Arya",
    license="Apache-2.0",
    # python_requires=">=3.12.8",
    install_requires=requirements,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "vlj=model.main:cli",
        ],
    },
)