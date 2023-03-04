#!/usr/bin/env python
# coding=utf-8
'''
File Description:
Author          : CHEN, JIA-LONG
Create Date     : 2023-02-11 13:50
FilePath        : \\N_20221213_ami_data_analyze\\Y_20230211_main_estimate_meter.py
Copyright © 2023 CHEN JIA-LONG.

將電表數據拉到全部，刪除太多和太少度數的用戶後，並改成7月或8月資料，
進行第一期期中報告的估測模型，對總用電量分群，找到該群裡面中心曲線最接近的用戶，
直接拿該用戶視為中心復原的根據，針對出來的資料，進行交叉驗證。
只有估測電表
'''
import copy
import logging
import os
import sys
from typing import TYPE_CHECKING, Iterable, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import ndarray
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
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
    user_data: DataFrame = indata.open_user_detail()
    user_data.out.log(name="user_data", logger=logging.debug, savecsv=False)
    trans_data: DataFrame = indata.open_transformer_data()
    trans_data.out.log(name="transformer data", logger=logging.debug, savecsv=False)
    # data_july: DataFrame = indata.open_convert_file(merge_how="inner")  # May data
    data: DataFrame = indata.open_6_12_file()
    data_july: DataFrame = data.meter.choose_timerange(
        start_date="2022-07-01", end_date="2022-08-01"
    )  # remain about 430 user.
    data_july = data_july.sort_values(by=['cust_id', 'meter_id', 'read_time'])
    data_july.reset_index(drop=True, inplace=True)
    data_july.out.log(name="July", logger=logging.debug)
    pre_data_jul: DataFrame = data_july.meter.preprocess(logger=None)  # remain 365 user.
    pre_data_jul = pre_data_jul.meter.delete_over_value(1000)
    pre_data_jul = pre_data_jul.meter.delete_below_value(31 * 2)  # remain 229 user.
    pre_data_jul.out.log(name="preprocess July", logger=logging.debug, savecsv=False)
    delete_meter: list = list(
        set(data_july.groupby(["cust_id"]).groups.keys())
        - set(pre_data_jul.groupby(["cust_id"]).groups.keys())
    )
    delete_data: DataFrame = data_july.set_index(["cust_id"]).loc[delete_meter]
    delete_data.reset_index(inplace=True)
    delete_data.out.log(name="delete data", logger=logging.debug)

    accuracy: list = []
    for count in range(1, 3):
        out.generate_root(f"{count}")
        x_train: DataFrame
        x_test: DataFrame
        y_train: DataFrame
        y_test: DataFrame
        x_train, x_test, y_train, y_test = pre_data_jul.meter.shuffle_split(
            test_size=0.2, random=True
        )
        x_train.out.log(name="x_train")
        x_test.out.log(name="x_test")

        sum_data: ndarray
        all_meter_data: DataFrame
        sum_data, all_meter_data = y_train.meter.kmeans_preprocess_total
        eblow: int = km.SSE(cluster_max=20, data=sum_data)
        km_module: KMeans = km.kmeans(
            n_clusters=eblow,
            summary_all_meter=all_meter_data.meter.summary_all_meter,
            data=sum_data,
            logger=logging.debug,
        )
        # km.draw_km_result_summary(data=y_train, km_module=km_module)
        estimate_module: DataFrame = km.estimate_module(
            data=y_train, km_module=km_module, draw=False
        )
        estimate_module.out.log(name="estimate", logger=logging.debug)
        y_test_estimate: DataFrame = km.estimate(
            x_test=x_test, y_train=y_train, km_module=km_module, estimate_module=estimate_module
        )
        y_test_estimate.set_index(["meter_id"], inplace=True)
        for meter in y_test_estimate.groupby(["meter_id"]).groups.keys():
            y_test_estimate.loc[meter, "cust_id"] = (
                y_test.set_index(["meter_id"]).loc[meter, "cust_id"].astype("string")
            )
        y_test_estimate.reset_index(inplace=True)
        y_test_estimate.out.log(name="y_test_estimate", logger=logging.debug)

        # km.estimate_check_currect(correct=y_test, estimate=y_test_estimate)
        acc: float = km.estimate_accurate(correct=y_test, estimate=y_test_estimate, draw=False)
        accuracy.append(acc)
        logging.debug(acc)
        print(acc)

    logging.debug(f"Every time accuracy = {accuracy}")
    logging.debug(f"Accuracy mean = {sum(accuracy)/len(accuracy)}")


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
            os.path.join(gl.get_value("OUTPUTLOGPATH"), "estimate_meter.log"),
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
