# Claude.md

I'm editing this from the prior edition; this project has a couple threads that have kind of superpoistioned within @fusionLearning/models. The first part of this project was building base segmentation models trained on known, standard datasets, with the purpose of then experimenting with attention methods for best fusion practices.

This thread is expired / no longer of interest to this study. Now, the main work is on the following: training base classification models on medical imaging datasets; these are harder to get, harder to do well on, and harder to interpret. With this in mind, the effort now is to, with the very simple & kind of well known hypothesis that ensembling multiple models can improve performance, build ensembles of classification models on medical imaging datasets.

The first dataset of this type is TOMPEI-CMMD. This dataset is a collection of medical imaging scans of breast cancer patients, with the goal of training a classification model to predict the presence or absence of LESIONS, not necessarily cancer. 

For this dataset, and any classification done onwards, any .py file withing models/ that ends in _cls.py is a MASTER script that is model, hyperparameter, and (soon) dataset agnostic, with the goal of having a one liner command training any model on any dataset, evaluating it, testing it, and producing all the necessary / desired graphs / post hoc analyses.

A very central goal in this whole research project is to develop sufficient empirical evidence, and displaying it in the right way, such that the hypothesis being tested is very foundationally evidenced; in the manuscript we'll be working on more statistical theory, and experiments that display the _exact_ phenomena we expect to see is very important.

## hardware

I will only be working on this within my remote environment, which has 4 L40S GPUs. For now, this is the only project that will be using these GPUs. The CUDA  version is 12.4 & driver version is 550.90.12

## project management / code style

Always display good codestyle; typehinting, docstrings (and class interfaces when needed) are welcome.
Be modular; sometimes I do an ad hoc experiment, and later i'll want to do something very similar but different, if you know what I mean. *You've been pretty good at this so far! keep it up*.

dont feel the need to write a lot of boilerplate, and you can mix it up when writing functional / oop code

Use uv for package management; 

uv add <library>
uv run <abc.py>
...

> never initiate training runs yourself / always ask for explicit permission; had a moment when server went down so no more full autonomy.
