from setuptools import setup, find_packages
import os

# Read the README.md file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="fusion-learning",
    version="0.1.0",
    author="crilillo14",
    author_email="",
    description="A machine learning framework for combining different models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/crilillo14/fusionLearning",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    package_data={
        'fusion_learning': [
            'weights/*',
            'config/*',
            'data/*'
        ]
    },
    include_package_data=True,
)
