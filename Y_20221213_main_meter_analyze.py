#!/usr/bin/env python
# coding=utf-8
"""
File Description: data analyze_依據一個月總用電度數進行分群，並找出總用電量與資料間的關係
Author          : CHEN, JIA-LONG
Create Date     : 2022-12-30 20:46
FilePath        : \\N_20221213_ami_data_analyze\\N_20221213_main.py
Copyright © 2023 CHEN JIA-LONG.
"""

import copy
import logging
import os
import sys
from typing import TYPE_CHECKING, Iterable, Tuple, Union

import pandas as pd
from numpy import ndarray
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))

import mylib.globalvar as gl
import mylib.module_InputDataV1 as indata
import mylib.module_kmeansV1 as km
import mylib.module_KShape as ks
import mylib.module_OutputData as out
from mylib.pandas_meterdataclass_extensions import ErrorAccessor, MeterDataAccessor, OutAccessor
from mylib.timer import Timer


@Timer(text="main used time: {:0.4f} seconds", logger=print)
@Timer(text="main used time: {:0.4f} seconds", logger=logging.debug)
def main() -> None:
    maininit()
    data: DataFrame = indata.open_convert_file(merge_how="inner")
    data.meter.log(name="Meter merge", logger=logging.debug)
    # indata.check_merge()

    """data preprocess"""
    preprocess_data: DataFrame = data.meter.preprocess(logger=None)
    datas_duplicated: DataFrame = preprocess_data.meter.col_duplicated()
    if not datas_duplicated.empty:
        datas_duplicated.meter.log(name="Duplicated(meter_id)", logger=logging.debug)

    """Delete TL18716098 meter"""
    preprocess_data.meter.del_specific_data(meter_id="TL18716098", inplace=True)

    """Meter value normalize"""
    normalize_data: DataFrame = preprocess_data.meter.normalize
    preprocess_data.meter.log(name="data preprocessed", logger=logging.debug)
    normalize_data.meter.log(name="data normalized", logger=logging.debug)

    """Calculate week_compare MAPE(only night, 1hr)"""
    # mape: pd.Series
    # mape_meter: pd.Series
    # mape, mape_meter = normalize_data.error.mape

    """k-means preprocess_total"""
    pre_kmeans_data: DataFrame
    data_kmeans_preprocess_summary: ndarray
    (
        data_kmeans_preprocess_summary,
        pre_kmeans_data,
    ) = normalize_data.meter.kmeans_preprocess_total
    pre_kmeans_data.meter.log(name="How week", logger=logging.debug, savecsv=False)

    """preprocess_1HR"""
    # pre_kmeans_1HR_data: DataFrame = pre_kmeans_data.meter.preprocess_1HR
    # pre_kmeans_1HR_data.meter.log(name="preprocess_1HR", logger=logging.debug)

    """kmeans"""
    # Consider delete over 1000 kwh meter.
    km.SSE(cluster_max=15, data=data_kmeans_preprocess_summary, figname="original_SSE")
    pre_kmeans_data_summary: Series = pre_kmeans_data.meter.summary_all_meter
    pre_kmeans_data_summary.meter.log(name="summary meter_id", logger=logging.info)
    kmeans_model: KMeans = km.kmeans(
        n_clusters=6,
        data=data_kmeans_preprocess_summary,
        summary_all_meter=pre_kmeans_data_summary,
        logger=None,
    )
    """Delete specific group kmeans, afresh kmeans."""
    # delete_keans_model: KMeans
    # fix_data_kmeans: Series
    # delete_keans_model, fix_data_kmeans = km.delete_group_kmeans(
    #     n_cluster=6,
    #     delete_group=[1, 3, 4],
    #     kmeans_model=kmeans_model,
    #     summary_all_meter=pre_kmeans_data_summary,
    #     logger=logging.debug,
    # )
    # km.SSE(cluster_max=15, data=fix_data_kmeans.values.reshape(-1, 1), figname="delete_SSE")

    """Draw all meter week compare picture."""
    # out.draw_week_compare(datas=preprocess_data)
    # out.all_meter_draw(datas_draw=pre_kmeans_data)
    # out.all_meter_draw(
    #     datas_draw=pre_kmeans_data, date=["2022-05-01 23:30:00", "2022-05-02 23:58:00"]
    # )
    """Try k-shape, k-means"""
    preprocess_data: DataFrame = preprocess_data.meter.delete_over_value(1000)
    # ks.SSE(cluster_max=100, data=preprocess_data.meter.week_value_15min, figname="K-shape SSE")
    # week_value: DataFrame = preprocess_data.meter.week_value_15min
    # week_value.out.log(name="week_value", logger=logging.debug, savecsv=True)
    # temp = week_value.to_numpy()
    # scaler: MinMaxScaler = MinMaxScaler(feature_range=(0.1, 0.9))
    # x_train: ndarray = scaler.fit_transform(temp)
    # # km.SSE(cluster_max=200, data=x_train, figname="SSE")
    # kmeans_model: KMeans = km.kmeans_common(
    #     n_clusters=35,
    #     data=x_train,
    # )
    # ks.kshape(data=temp, n_clusters=35)

    """delete the value above specific value."""
    # preprocess_data.out.log(name="delete over 1000", logger=logging.debug)
    # week_value: DataFrame = preprocess_data.meter.week_value
    # week_value.out.log(name="week_value", logger=logging.debug)
    # # week_night_value: DataFrame = preprocess_data.meter.week_night
    # # week_night_value.out.log(name="week_night_value", logger=logging.debug)
    # week_value_15min: DataFrame = preprocess_data.meter.week_value_15min
    # week_value_15min.out.log(name="week_15min_value", logger=logging.debug)

    out.draw_week_compare(
        datas=preprocess_data,
        dayhavedata=96,
        xlabel="time(week_15min)",
        ylabel="kwh/15min",
        time="15min",
        drawrange="7day",
    )
    # ks.SSE(cluster_max=10, data=preprocess_data.meter.week_value_15min, figname="K-shape SSE")


def maininit() -> None:
    """main program initialize."""
    os.makedirs(os.path.join(os.getcwd(), "output"), exist_ok=True)
    gl.set_value("OUTPUTLOGPATH", os.path.join(os.getcwd(), "output"), readonly=True)
    gl.set_value("CODEFILE", os.path.splitext(os.path.basename(__file__))[0], readonly=True)
    gl.set_value("INPUTPATH", os.path.join(os.getcwd(), "input"), readonly=True)
    # logname: str = input("Enter the Log file name, otherwise press Enter to skip:")
    logname: str = ""
    if logname == "":
        file_handler_args: Tuple = (
            os.path.join(gl.get_value("OUTPUTLOGPATH"), "meter_analyze.log"),
            "w",
            "utf-8",
        )
    else:
        file_handler_args: Tuple = (
            os.path.join(gl.get_value("OUTPUTLOGPATH"), f"{logname}.log"),
            "w",
            "utf-8",
        )
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(*file_handler_args)],
    )
    logging.getLogger("matplotlib.font_manager").disabled = True
    logging.getLogger("PIL").setLevel(logging.WARNING)


if __name__ == "__main__":
    if TYPE_CHECKING:  # Using for visual studio code type hint.

        class DataFrame(pd.DataFrame):
            meter: MeterDataAccessor
            error: ErrorAccessor
            out: OutAccessor

        class Series(pd.Series):
            meter: MeterDataAccessor
            out: OutAccessor

    gl._init()  # global python file need initialize before main program.
    main()
