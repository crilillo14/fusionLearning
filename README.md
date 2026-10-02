Everything is a bit of a mess right now, but this repository contains experiments related to stable & robust ensembling of deep classifiers & segmentation models.

@fusionLearning/models/ contains segmentation & classification model training and any post hoc experiments done. Figures & results stored in models/results/{Dataset}/.

data ingestion, transformation & distributed sampling is done in @fusionLearning/data/. 

Ensemble "stability" is multi-faceted, and the variety of post hoc experiments reflect that. For example, modern takes on the bias variance decompositions ([Belkin2019](https://arxiv.org/abs/1812.11118), [Yang2019](https://proceedings.mlr.press/v119/yang20j)[Gupta2022](https://arxiv.org/abs/2206.10566)) of overparametrized models challenge the classical bias-variance tradeoff believed to be true for much of the 20th and 21st century (this phenomenon is often called "double descent"). Because of this, if you want to argue a minima exists for the generalized error, you have to first show that models you're ensembling aren't undergoing double descent. Under this scenario, which is commonly induced in data scarce learning tasks, arguing that ensembling is _optimal_ follows cleanly from g-Bregman bias-var decompositions ([Gupta2022](https://arxiv.org/abs/2206.10566)), see also Pfau2013.

Most datasets used are _clinical_, such as lesion detection in mammographies. Because labelling is expensive, and lesion observations are few and far between, scaling model size & compute (as Chinchilla scaling laws suggest) will not have the intended consequence, rather inducing overfitting or model collapse. Despite this, clinical classification models have improved relentlessly, levaraging small accruing heuristics to overcome data sparsity. 

Robustness requires many experiments to parametrize, but essentially boils down to introducing synthetic errors with photometric data transforms. 

Once training, logging, graphing, & housekeeping is done, I'll be sure to include a short exposition of the results.
