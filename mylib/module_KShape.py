#!/usr/bin/env python
# coding=utf-8
'''
File Description:
Author          : CHEN, JIA-LONG
Create Date     : 2023-02-07 09:54
FilePath        : \\N_20221213_ami_data_analyze\\mylib\\module_KShape.py
Copyright © 2023 CHEN JIA-LONG.
'''
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numpy import ndarray
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from .module_OutputData import FigPara


def kshape(data: ndarray, n_clusters: int, verbose: bool = True, random_seed: int = 42):
    x_train: ndarray = TimeSeriesScalerMeanVariance(mu=1, std=1).fit_transform(data)
    # x_train: ndarray = TimeSeriesScalerMeanVariance().fit_transform(data)
    x_limit: list[int] = [0, x_train.shape[1]]
    ks: KShape = KShape(n_clusters=n_clusters, verbose=verbose, random_state=random_seed)
    y_pred: ndarray = ks.fit_predict(x_train)

    for yi in range(n_clusters):
        with FigPara(
            title=f"KShape_Cluster {yi+1}",
            show=False,
            save=True,
            generate_newfile=f"KShape_{n_clusters}",
            xlim=x_limit,
        ) as f:
            for xx in x_train[y_pred == yi]:
                plt.plot(xx.ravel(), "k-", alpha=0.2)
            plt.plot(ks.cluster_centers_[yi].ravel(), "r-")


def SSE(cluster_max: int, data: ndarray, random: int = 24, figname: str = "SSE") -> None:
    """Draw and save KShape's SSE picture.

    Args:
        cluster_max (int): Max group number
        data (array): Need to cluster analysis data.
        random (int, optional): KMeans random parament. Defaults to 1.
    """
    distortions: list = []
    x_train = TimeSeriesScalerMeanVariance(mu=1, std=1).fit_transform(data)
    for i in range(2, cluster_max + 1):
        ks: KShape = KShape(n_clusters=i, verbose=True, random_state=random).fit(x_train)
        distortions.append(ks.inertia_)  # 誤差平方和(SSE)
    with FigPara(xlabel="k", ylabel="SSE", linewidth=3, title=figname) as f:
        plt.plot(range(2, cluster_max + 1), distortions, linewidth=f.linewidth)
