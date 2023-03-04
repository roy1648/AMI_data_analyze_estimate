#!/usr/bin/env python
# coding=utf-8
"""
File Description: Kmeans module file.
Author          : CHEN, JIA-LONG
Create Date     : 2022-12-30 20:46
FilePath        : \\N_20221213_ami_data_analyze\\module_kmeansV1.py
Copyright © 2023 CHEN JIA-LONG.
"""

import array
import copy
import logging
import os
from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from kneed import KneeLocator
from matplotlib.figure import Figure
from numpy import ndarray
from pandas.core.frame import DataFrame
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.series import Series
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from . import globalvar as gl
from .module_OutputData import FigPara, generate_outfile
from .pandas_meterdataclass_extensions import ErrorAccessor, MeterDataAccessor, OutAccessor

if TYPE_CHECKING:  # Using for visual studio code type hint.

    class DataFrame(pd.DataFrame):
        meter: MeterDataAccessor
        error: ErrorAccessor
        out: OutAccessor

    class Series(pd.Series):
        meter: MeterDataAccessor
        out: OutAccessor


class NoMatchError(Exception):
    """If no match two data, will raise this error"""


def kmeans_common(
    n_clusters: int,
    data: np.ndarray,
    savename: str = "kmeans_model",
    random: int = 42,
    logger: Optional[Callable[[str], None]] = logging.debug,
):
    # scaler = MinMaxScaler(feature_range=(0.1, 0.9))
    # x_train: ndarray = scaler.fit_transform(data)
    # x_train: ndarray = TimeSeriesScalerMeanVariance(mu=1, std=1).fit_transform(data)
    # x_train = x_train.reshape(x_train.shape[0], (x_train.shape[1] * x_train.shape[2]))
    x_train = data
    x_limit: list[int] = [0, x_train.shape[1]]
    kmeans_fit: KMeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=random).fit(
        x_train
    )
    y_pred: ndarray = kmeans_fit.fit_predict(x_train)
    generate_outfile()
    joblib.dump(value=kmeans_fit, filename=f"{gl.get_value('OUTPUTPATH')}\\{savename}")

    for yi in range(n_clusters):
        with FigPara(
            title=f"Kmeans_Cluster {yi+1}",
            show=False,
            save=True,
            generate_newfile=f"Kmeans_{n_clusters}",
            xlim=x_limit,
        ) as f:
            for xx in x_train[y_pred == yi]:
                plt.plot(xx.ravel(), "k-", alpha=0.2)
            plt.plot(kmeans_fit.cluster_centers_[yi].ravel(), "r-")

    return joblib.load(f"{gl.get_value('OUTPUTPATH')}\\{savename}")


def kmeans(
    n_clusters: int,
    summary_all_meter: Series,
    data: np.ndarray,
    savename: str = "kmeans_model",
    random: int = 42,
    logger: Optional[Callable[[str], None]] = logging.debug,
) -> KMeans:
    """KMeans Cluster Analysis data and store result.

    Args:
        clusters (int): Analyze the number of groups.
        data (array): Need to cluster analysis data.
        savename (str, optional): _description_. Defaults to "kmeans_model".
        random (int, optional): _description_. Defaults to 1.
        logger (Optional[str], optional): _description_. Defaults to logging.debug.
    """
    kmeans_fit: KMeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=random).fit(
        data
    )
    generate_outfile()
    joblib.dump(value=kmeans_fit, filename=f"{gl.get_value('OUTPUTPATH')}\\{savename}")
    if logger is not None:
        kmeans_loaded_model: KMeans = joblib.load(f"{gl.get_value('OUTPUTPATH')}\\{savename}")
        labels: np.ndarray = kmeans_loaded_model.labels_
        for i in np.unique(labels):
            logger(f'{i+1} group have: {summary_all_meter.index[labels == i].size} meter.')  # 電表有幾個
            logger(f'{i+1} group: \n{summary_all_meter.index[labels == i].values}')  # 此群有那些電表
        centers: array = kmeans_loaded_model.cluster_centers_
        for i, j in enumerate(centers):
            logger(f"{i+1} group cesnter is: {j[0]:.1f} kwh.")
    return joblib.load(f"{gl.get_value('OUTPUTPATH')}\\{savename}")


def delete_group_kmeans(
    n_cluster: int,
    delete_group: list,
    kmeans_model: KMeans,
    summary_all_meter: Series,
    savename: str = "delete_kmeans_model",
    logger: Optional[Callable[[str], None]] = logging.debug,
) -> Tuple[KMeans, Series]:
    labels: np.ndarray = kmeans_model.labels_
    delete_AMI: np.ndarray = np.array([])
    for i in delete_group:
        delete_AMI = np.append(delete_AMI, summary_all_meter.index[labels == i].values)
    if logger is not None:
        logger(f'Delete AMI:{delete_AMI}')
    fix_data_kmeans: Series = summary_all_meter[~summary_all_meter.index.isin(delete_AMI)]

    new_kmeans_model: KMeans = kmeans(
        n_clusters=n_cluster,
        summary_all_meter=fix_data_kmeans,
        data=fix_data_kmeans.values.reshape(-1, 1),
        savename=savename,
        logger=logger,
    )
    return new_kmeans_model, fix_data_kmeans


def SSE(
    cluster_max: int,
    data: ndarray,
    random: int = 42,
    figname: str = "SSE",
    save_path: Optional[str] = None,
) -> int:
    """Draw and save KMeans's SEE picture.

    Args:
        cluster_max (int): Max group number
        data (array): Need to cluster analysis data.
        random (int, optional): KMeans random parament. Defaults to 1.
    """
    distortions: list = []
    # scores: list = []
    for i in range(2, cluster_max + 1):
        kmeans: KMeans = KMeans(n_clusters=i, init='k-means++', random_state=random).fit(data)
        distortions.append(kmeans.inertia_)  # 誤差平方和(SSE)
    #     scores.append(silhouette_score(data, kmeans.predict(data)))  # 側影係數
    # selected_K: int = scores.index(max(scores)) + 2
    knee: KneeLocator = KneeLocator(
        range(2, cluster_max + 1), distortions, curve="convex", direction="decreasing"
    )
    generate_outfile()
    knee.plot_knee(title="SSE", xlabel="Number of Clusters", ylabel="SSE")
    f: FigPara = FigPara()
    if save_path:
        path: str = os.path.join(save_path, "SSE.jpg")
    else:
        path: str = os.path.join(gl.get_value("OUTPUTPATH"), "SSE.jpg")
    plt.savefig(
        path,
        dpi=f.dpi,
        bbox_inches=f.bbox_inches,
        pad_inches=f.pad_inches,
    )
    plt.clf()
    plt.close()
    return knee.elbow


def draw_km_result_summary(data: pd.DataFrame, km_module: KMeans) -> None:
    center: ndarray = km_module.cluster_centers_
    center_sort: ndarray = np.sort(center.reshape(-1))
    labels: ndarray = km_module.labels_
    sum: Series = data.meter.summary_all_meter.values.reshape(-1, 1)
    generate_outfile()
    with FigPara(title="K_Means result", ylabel="kwh") as f:
        for i in range(0, center_sort.size):
            value: Series = sum[labels == np.where(center == center_sort[i])[0][0]]
            plt.plot(np.full_like(value, i + 1), value, 'x', label=i + 1, markersize=f.markersize)
        plt.legend(prop={'size': f.legend_size})


def estimate_module(
    data: pd.DataFrame,
    km_module: KMeans,
    draw: bool = False,
    logger: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    label: ndarray = np.unique(km_module.labels_)
    sum: Series = data.meter.summary_all_meter
    copy_data: DataFrame = copy.deepcopy(data)
    size: int = data.groupby(["read_time"]).ngroups
    copy_data.set_index(['meter_id', 'cust_id', 'read_time'], inplace=True)
    copy_data.sort_index(level=['meter_id', 'cust_id', 'read_time'], inplace=True)
    time_index: dict = (
        copy_data.index.get_level_values('read_time')
        .groupby(copy_data.index.get_level_values('read_time'))
        .keys()
    )
    center_15: DataFrame = pd.DataFrame(index=time_index)
    copy_data.insert(len(copy_data.columns), "center_15", np.nan)
    scaler: MinMaxScaler = MinMaxScaler(feature_range=(0.1, 0.9))
    for i in label:
        decrease_count: int = 0
        center_15sum: ndarray = np.zeros(shape=size)
        for meter in sum.index[km_module.labels_ == i]:
            if np.any(np.isnan(copy_data.loc[meter, 'ratio_kwh'])):
                decrease_count += 1
                nan_bool: ndarray = np.isnan(copy_data.loc[meter, 'center_15'])
                raise ValueError("Ratio*kwh have none value.")
            scalered: ndarray = scaler.fit_transform(
                copy_data.loc[meter, 'ratio_kwh'].values.reshape(-1, 1)
            )
            center_15sum = center_15sum + scalered.reshape(-1)
            # center_15sum = center_15sum + copy_data.loc[meter, 'ratio_kwh'].values
        center_15sum = center_15sum / (sum.index[km_module.labels_ == i].size - decrease_count)
        center_15.insert(len(center_15.columns), f"center_{i}", center_15sum)
        if logger:
            logger(f"Center_{i+1} = {center_15sum.sum()}")
    estimate_center_meter: DataFrame = pd.DataFrame(index=label)
    for i in label:
        dis: DataFrame = pd.DataFrame(index=sum.index[km_module.labels_ == i])
        for meter in sum.index[km_module.labels_ == i]:
            kwh: ndarray = scaler.fit_transform(
                copy_data.loc[meter, 'ratio_kwh'].values.reshape(-1, 1)
            )
            cen: ndarray = scaler.fit_transform(center_15[f"center_{i}"].values.reshape(-1, 1))
            dis.loc[meter, 'distance'] = distance.euclidean(kwh, cen)
        estimate_center_meter.loc[i, 'meter_id'] = dis.idxmin().distance
        estimate_center_meter.loc[i, 'Euclidean_distance'] = dis.min().values
        estimate_center_meter.loc[i, 'index'] = f"center_{i}"
    estimate_center_meter.set_index('index', inplace=True)
    if logger:
        estimate_center_meter.out.log(name="estimate_meter", logger=logging.debug)

    if draw:
        with FigPara(
            title="Center_estimate", ylabel="kwh", xlabel="time", generate_newfile="Center_estimate"
        ) as f:
            time: list = list(center_15.index)
            drawline: list = []
            pic_label: list = []
            for i in center_15.columns:
                drawline.append(time)
                drawline.append(center_15[i])
                pic_label.append(f"{i}")
            pl: list = plt.plot(*drawline, linewidth=f.linewidth)
            plt.legend(pl, pic_label, prop={"size": f.legend_size}, loc="upper right")

        for j in center_15.columns:
            with FigPara(
                title=f"{j}", ylabel="kwh", xlabel="time", generate_newfile="Center_estimate"
            ) as f:
                time: list = list(center_15.index)
                m: str = estimate_center_meter.loc[j, 'meter_id']
                plt.plot(
                    time,
                    copy_data.loc[m, 'ratio_kwh'].values,
                    'k-',
                    alpha=0.3,
                    label=m,
                    linewidth=f.linewidth,
                )
                plt.plot(time, center_15[j], label="center", linewidth=f.linewidth)
                plt.legend(prop={"size": f.legend_size}, loc="upper right")
    return estimate_center_meter


def estimate(
    x_test: pd.Series, y_train: pd.DataFrame, km_module: KMeans, estimate_module: DataFrame
) -> pd.DataFrame:
    cluster_labels: ndarray = km_module.predict(x_test.values.reshape(-1, 1))
    x_test_frame: DataFrame = x_test.to_frame()
    x_test_frame.insert(len(x_test_frame.columns), "label", cluster_labels)
    x_test_frame.reset_index(inplace=True)
    x_test_frame.set_index(["label", "meter_id"], inplace=True)
    estimate_module_reset: DataFrame = estimate_module.reset_index()
    train_data: DataFrame = y_train.set_index("meter_id", inplace=False)
    train_sum: Series = y_train.meter.summary_all_meter
    index1: list = list(x_test_frame.index.get_level_values(level="meter_id"))
    index2: list = list(y_train.set_index(["read_time"]).groupby(["read_time"]).groups.keys())
    index_iterables: list = [index1, index2]
    index: pd.MultiIndex = pd.MultiIndex.from_product(
        index_iterables, names=["meter_id", "read_time"]
    )
    result: DataFrame = pd.DataFrame(index=index, columns=["ratio_kwh"])
    idx = pd.IndexSlice
    for label in np.unique(cluster_labels):
        meter: str = estimate_module_reset.loc[label, "meter_id"]
        center_kwh: DataFrame = train_data.loc[meter, ["read_time", "ratio_kwh"]]
        center_kwh.sort_values(by=["read_time"], inplace=True)
        center_sum: int = train_sum[meter]
        x_test_sum: DataFrame = x_test_frame.loc[label]
        for m in x_test_sum.index:
            if np.any(~(result.loc[m, "ratio_kwh"].index == center_kwh["read_time"])):
                raise NoMatchError("read_time can't be aligned.")
            else:
                result.loc[idx[m, :], "ratio_kwh"] = list(
                    center_kwh['ratio_kwh'] * x_test_sum.loc[m, 'ratio_kwh'] / center_sum
                )
    result.reset_index(inplace=True)
    return result


def estimate_check_currect(correct: pd.DataFrame, estimate: pd.DataFrame) -> None:
    sum_estimate: Series = estimate.meter.summary_all_meter
    sum: Series = correct.meter.summary_all_meter
    with FigPara(title="check estimate correct", markersize=20, figsize=[3, 40]) as f:
        plt.plot(
            np.full_like(sum.values, 1),
            sum.values,
            'x',
            label="original",
            markersize=f.markersize,
        )
        plt.plot(
            np.full_like(sum_estimate.values, 2),
            sum_estimate.values,
            'x',
            label="estimate",
            markersize=f.markersize,
        )
        plt.legend(prop={'size': f.legend_size})
        f.fig.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
        # f.fig.gca().yaxis.set_major_locator(ticker.MultipleLocator(50))


def estimate_accurate(correct: pd.DataFrame, estimate: pd.DataFrame, draw: bool = False) -> float:
    correct_fill0: DataFrame = (
        correct.set_index(["read_time"]).groupby("meter_id").resample('15min').sum().reset_index()
    )  # Resample the time axis to every 15 minutes and fill missing data with 0.

    time: list = list(estimate.groupby(["read_time"]).groups.keys())
    time.sort()
    empty_data: DataFrame = pd.DataFrame(0, index=pd.Index(time), columns=["ratio_kwh"])
    empty_data.index.name = "read_time"
    empty_data.reset_index(inplace=True)
    correct_index: DataFrame = correct.set_index(["meter_id", "read_time"], inplace=False)
    correct_index_fill0: DataFrame = correct_fill0.set_index(
        ["meter_id", "read_time"], inplace=False
    )
    meter: list = list(correct.groupby(["meter_id"]).groups.keys())
    time: list = list(correct.groupby(["read_time"]).groups.keys())
    estimate_index: DataFrame = estimate.set_index(["meter_id", "read_time"], inplace=False)
    idx = pd.IndexSlice
    acc: float = 0
    for m in meter:
        havevalue: bool = False
        correct_read_time: DatetimeIndex = correct_index.loc[
            idx[m, :], "ratio_kwh"
        ].index.get_level_values(level="read_time")
        estimate_read_time: DatetimeIndex = estimate_index.loc[
            idx[m, :], "ratio_kwh"
        ].index.get_level_values(level="read_time")
        try:
            if np.any(~(correct_read_time == estimate_read_time)):
                raise NoMatchError("Read time can't be aligned.")
            else:
                cor: ndarray = correct_index.loc[idx[m, :], "ratio_kwh"].values.reshape(-1, 1)
        except ValueError:
            logging.error(
                f"Meter id {m} have {correct_read_time.size} data points"
                + f", but {estimate_read_time.size} data points are required"
            )
            cor: ndarray = correct_index_fill0.loc[idx[m, :], "ratio_kwh"].values.reshape(-1, 1)
            if cor.size != estimate.size:
                value: DataFrame = (
                    pd.concat(
                        [
                            correct_index_fill0.loc[idx[m, :], "ratio_kwh"]
                            .reset_index()
                            .drop('meter_id', axis=1),
                            empty_data,
                        ]
                    )
                    .groupby("read_time")
                    .sum()
                    .sort_index()
                    .reset_index()
                )
                havevalue = True
                cor = value.set_index("read_time").values.reshape(-1, 1)

        est: ndarray = estimate_index.loc[idx[m, :], "ratio_kwh"].values.reshape(-1, 1)
        dis: float = distance.euclidean(cor, est)
        acc += dis
        if draw:
            with FigPara(
                title=f"{m}_{dis:.1f}", xlabel="time", ylabel="kwh", logger=logging.debug
            ) as f:
                if havevalue:
                    plt.plot(
                        time,
                        value["ratio_kwh"].values,
                        linewidth=f.linewidth,
                        label="correct",
                    )
                else:
                    plt.plot(
                        time,
                        correct_index_fill0.loc[idx[m, :], "ratio_kwh"].values,
                        linewidth=f.linewidth,
                        label="correct",
                    )
                plt.plot(
                    time,
                    estimate_index.loc[idx[m, :], "ratio_kwh"].values,
                    linewidth=f.linewidth,
                    label="estimate",
                )
                plt.legend(prop={"size": f.legend_size})

    return acc / len(meter)


def estimate_acc_trans_uti_rate(
    coor_group: list,
    correct: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]],
    estimate: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]],
    draw: bool = False,
) -> float:
    def _mape(c: pd.DataFrame, e: pd.DataFrame, t: str) -> float:
        c_index: DataFrame = c.sort_index()
        e_index: DataFrame = e.sort_index()
        time: list = list(c_index.reset_index().groupby(["index"]).groups.keys())
        _mape: Series = np.mean(abs(e_index - c_index) / c_index) * 100
        if draw:
            with FigPara(
                title=f"{t}_{_mape.rate:.1f}%",
                xlabel="time",
                ylabel="utilization_rate(%)",
                logger=logging.debug,
            ) as f:
                plt.plot(
                    time,
                    c_index.values,
                    linewidth=f.linewidth,
                    label="correct",
                )
                plt.plot(
                    time,
                    e_index.values,
                    linewidth=f.linewidth,
                    label="estimate",
                )
                plt.legend(prop={"size": f.legend_size})
        return _mape.rate

    if type(correct) is tuple and type(estimate) is tuple:
        correct_resident: DataFrame = correct[0]
        correct_industry: DataFrame = correct[1]
        estimate_resident: DataFrame = estimate[0]
        estimate_industry: DataFrame = estimate[0]
        mape_resident: float = _mape(
            c=correct_resident, e=estimate_resident, t=f"{coor_group}_resident"
        )
        mape_industry: float = _mape(
            c=correct_industry, e=estimate_industry, t=f"{coor_group}_industry"
        )
        # _mape_draw(title=f"{coor_group}_resident_{mape_resident:.1f}%")
        # _mape_draw(title=f"{coor_group}_industry_{mape_industry:.1f}%")
        return mape_resident, mape_industry
    else:
        mape: float = _mape(c=correct, e=estimate, t=f"{coor_group}")

        # correct.out.log(name="correct", logger=logging.debug)
        # estimate.out.log(name="estimate", logger=logging.debug)

        # correct_index: DataFrame = correct.sort_index()
        # # estimate_index: DataFrame = estimate.sort_index()

        # # mape: float = np.mean(abs(estimate_index - correct_index) / correct_index) * 100

        # # cor: ndarray = correct_index.values.reshape(-1, 1)
        # # est: ndarray = estimate_index.values.reshape(-1, 1)
        # # dis: float = distance.euclidean(cor, est)
        # time: list = list(correct_index.reset_index().groupby(["index"]).groups.keys())
        # if draw:
        #     with FigPara(
        #         title=f"{coor_group}_{mape.rate:.1f}%",
        #         xlabel="time",
        #         ylabel="utilization_rate(%)",
        #         logger=logging.debug,
        #     ) as f:
        #         plt.plot(
        #             time,
        #             correct_index.values,
        #             linewidth=f.linewidth,
        #             label="correct",
        #         )
        #         plt.plot(
        #             time,
        #             estimate_index.values,
        #             linewidth=f.linewidth,
        #             label="estimate",
        #         )
        #         plt.legend(prop={"size": f.legend_size})
        return mape
