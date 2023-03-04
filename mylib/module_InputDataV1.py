#!/usr/bin/env python
# coding=utf-8
"""
Author       : CHEN, JIA-LONG
Date         : 2022-12-30 11:01
LastEditTime : 2023-01-06 11:23
FilePath     : \\N_20221213_ami_data_analyze\\module_InputDataV1.py
Description  : Input Data code file.
Copyright © 2022 CHEN JIA-LONG.
"""
import glob
import logging
import os
import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from . import globalvar as gl
from .pandas_meterdataclass_extensions import OutAccessor
from .timer import Timer

if TYPE_CHECKING:  # Using for visual studio code type hint.

    class DataFrame(pd.DataFrame):
        out: OutAccessor

    class Series(pd.Series):
        out: OutAccessor


@Timer(text="open file used time: {:0.4f} seconds", logger=logging.debug)
def open_convert_file(merge_how: str = "inner") -> pd.DataFrame:
    """open and convert meter's files and users's file.
    匯入每15分鐘的子檔案，與用戶資料進行合併，留下這些欄位['圖號座標','組別','cust_id','meter_id','read_time', 'ratio_kwh']

    Args:
        merge_how (str, optional): Can select "outer" or "inner". Defaults to "inner".

    Returns:
        _padas.Dataframe_: merged ami data.
    """
    files_joined: str = os.path.join(
        gl.get_value("INPUTPATH"), "SL67_15minAMI", "lpi*.csv"
    )  # 設定抓取csv路徑
    # files_joined = os.path.join(path,"lpi_sl67_20220501.csv")   # 設定抓取csv路徑
    list_files: list = glob.glob(files_joined)  # 尋找此路徑下所有的資料
    datas: pd.DataFrame = pd.concat(
        map(pd.read_csv, list_files), ignore_index=True
    )  # 透過路徑尋找所有csv檔案整併
    files_joined = os.path.join(
        gl.get_value("INPUTPATH"), "SL67_15minAMI", "2022-06-08_用戶資料查詢(北市長春SL67).csv"
    )  # 設定抓取csv路徑
    datas_user: pd.DataFrame = pd.read_csv(files_joined, usecols=["圖號座標", "組別", "用戶電號"])
    # datas_user = datas_user.rename(columns={"id": "cust_id"})
    datas_user = datas_user.rename(columns={"用戶電號": "cust_id"})
    datas_user = pd.merge(datas_user, datas, on="cust_id", how=merge_how)  # how = outer or inner
    datas_user = datas_user.drop(axis=1, columns=["rec_kwh"])  # delete rec_kwh column
    """convert datas"""
    datas_user["read_time"] = datas_user["read_time"].map(
        {k: pd.to_datetime(k) for k in datas_user["read_time"].unique()}
    )
    datas_user["ratio_kwh"] = datas_user["ratio"] * datas_user["del_kwh"]  # conculate ratio*kwh
    datas_user.pop("ratio")
    datas_user.pop("del_kwh")
    datas_user["ratio_kwh"] = pd.to_numeric(
        datas_user["ratio_kwh"], downcast="float", errors="ignore"
    )  # 轉換成整數並且降低暫存容量
    datas_user["cust_id"] = pd.to_numeric(
        datas_user["cust_id"], downcast="integer", errors="ignore"
    )  # 轉換成整數並且降低暫存容量
    return datas_user


def check_merge(logger: str = logging.debug) -> tuple:
    """Check different data can all merge?

    Args:
        logger (str, optional): Select log method(logging.debug/logging.info/print).
                                Defaults to logging.debug.

    Returns:
        tuple: (coordinate can't match, meter_id can't match, kwh can't match)
    """
    datas: pd.DataFrame = open_convert_file(merge_how="outer")
    logger(datas.isnull().sum())
    datas_temp: pd.DataFrame = datas[datas.圖號座標.isnull()]
    logger(datas_temp)
    datas_temp_group: dict = datas_temp.groupby(["meter_id"]).groups.keys()
    logger(datas_temp_group)
    datas_kwh_null: pd.DataFrame = datas[datas.ratio_kwh.isnull()]
    logger(datas_kwh_null.to_string(index=False))
    return datas_temp, datas_temp_group, datas_kwh_null


@Timer(text="open file used time: {:0.4f} seconds", logger=logging.debug)
def open_6_12_file(merge_how: str = "inner") -> pd.DataFrame:
    """open and convert meter's files.
    Args:
        merge_how (str, optional): Can select "outer" or "inner". Defaults to "inner".

    Returns:
        _padas.Dataframe_: merged ami data.
    """
    sl67_path: str = os.path.join(gl.get_value("INPUTPATH"), "SL67_6_12.csv")  # 設定抓取csv路徑
    sl67_ratio_path: str = os.path.join(gl.get_value("INPUTPATH"), "sl67倍數.csv")  # 設定抓取csv路徑
    sl67_data: DataFrame = pd.read_csv(sl67_path)
    sl67_data.pop("rec_kwh")
    sl67_ratio: DataFrame = pd.read_csv(sl67_ratio_path)
    sl67_ratio["meter_id"] = sl67_ratio["新型模組化低壓AMI形式代號"].astype("str") + sl67_ratio["表號"].astype(
        "str"
    )
    sl67_ratio.pop("新型模組化低壓AMI形式代號")
    sl67_ratio.pop("表號")
    result: DataFrame = pd.merge(sl67_data, sl67_ratio, on="meter_id", how=merge_how)
    result.pop("cust_id")
    result["ratio_kwh"] = result["電表倍數"] * result["del_kwh"]  # conculate ratio*kwh
    result.pop("del_kwh")
    result.pop("電表倍數")
    result.rename(columns={"用戶電號": "cust_id"}, inplace=True)
    result = result[["cust_id", "meter_id", "read_time", "ratio_kwh"]]
    result["read_time"] = result["read_time"].map(
        {k: pd.to_datetime(k) for k in result["read_time"].unique()}
    )
    return result


@Timer(text="open user file used time: {:0.4f} seconds", logger=logging.debug)
def open_user_detail() -> pd.DataFrame:
    """
    Returns:
        _padas.Dataframe_: merged user data.
    """
    files_joined: str = os.path.join(
        gl.get_value("INPUTPATH"), "20221122提供用戶明細_csvutf8", "變壓器負載-用戶明細-*.csv"
    )  # 設定抓取csv路徑
    list_files: list = glob.glob(files_joined)  # 尋找此路徑下所有的資料
    datas: DataFrame = pd.concat(
        [
            pd.read_csv(f, encoding="utf-8", usecols=["表號", "相別", "契約容量", "變壓器座標"])
            for f in list_files
        ],
        ignore_index=True,
    )  # 透過路徑尋找所有csv檔案整併
    datas["表號"] = [re.sub("-", "", elem) for elem in datas["表號"].values]
    datas["表號"] = [re.sub("^00", "", elem) for elem in datas["表號"].values]
    datas["圖號座標"] = np.nan
    datas["組別"] = np.nan
    for i, v in enumerate(datas["變壓器座標"].values):
        coor_group: list = re.split("  ", v)
        datas.loc[i, ('圖號座標')] = coor_group[0]
        datas.loc[i, ('組別')] = re.sub("^S00", "", coor_group[1])
    datas.drop(columns=["變壓器座標"], inplace=True)
    return datas


@Timer(text="open transformer file used time: {:0.4f} seconds", logger=logging.debug)
def open_transformer_data() -> pd.DataFrame:
    """Open transformer data."""
    trans_path: str = os.path.join(gl.get_value("INPUTPATH"), "SL67-變壓器.csv")  # 設定抓取csv路徑
    col: list[str] = ["主設備圖號座標", "組別", "第一具變壓器容量", "第二具變壓器容量", "第三具變壓器容量", "用戶數"]
    data: DataFrame = pd.read_csv(trans_path, encoding="utf-8", usecols=col)
    return data


def open_mk33_may_haveratio(merge_how: str = "inner") -> pd.DataFrame:
    """open and convert meter's files and users's file.
    匯入每15分鐘的子檔案，與用戶資料進行合併，留下這些欄位['圖號座標','組別','cust_id','meter_id','read_time', 'ratio_kwh']

    Args:
        merge_how (str, optional): Can select "outer" or "inner". Defaults to "inner".

    Returns:
        _padas.Dataframe_: merged ami data.
    """
    files_joined: str = os.path.join(
        gl.get_value("INPUTPATH"), "mk33_15minAMI", "lpi*.csv"
    )  # 設定抓取csv路徑
    list_files: list = glob.glob(files_joined)  # 尋找此路徑下所有的資料
    datas: pd.DataFrame = pd.concat(
        map(pd.read_csv, list_files), ignore_index=True
    )  # 透過路徑尋找所有csv檔案整併
    files_joined = os.path.join(
        gl.get_value("INPUTPATH"), "mk33_15minAMI", "2022-06-08_用戶資料查詢 (高雄MK33).csv"
    )  # 設定抓取csv路徑
    datas_user: pd.DataFrame = pd.read_csv(files_joined, usecols=["圖號座標", "組別", "用戶電號"])
    datas_user = datas_user.rename(columns={"用戶電號": "cust_id"})
    datas_user = pd.merge(datas_user, datas, on="cust_id", how=merge_how)  # how = outer or inner
    datas_user = datas_user.drop(axis=1, columns=["rec_kwh"])  # delete rec_kwh column
    """convert datas"""
    datas_user["read_time"] = datas_user["read_time"].map(
        {k: pd.to_datetime(k) for k in datas_user["read_time"].unique()}
    )
    datas_user["ratio_kwh"] = datas_user["ratio"] * datas_user["del_kwh"]  # conculate ratio*kwh
    datas_user["ratio_kwh"] = pd.to_numeric(
        datas_user["ratio_kwh"], downcast="float", errors="ignore"
    )  # 轉換成整數並且降低暫存容量
    datas_user["cust_id"] = pd.to_numeric(
        datas_user["cust_id"], downcast="integer", errors="ignore"
    )  # 轉換成整數並且降低暫存容量
    return datas_user


@Timer(text="open mk33_may data used time: {:0.4f} seconds", logger=logging.debug)
def open_mk33_may(merge_how: str = "inner") -> pd.DataFrame:
    """open and convert meter's files and users's file.
    匯入每15分鐘的子檔案，與用戶資料進行合併，留下這些欄位['圖號座標','組別','cust_id','meter_id','read_time', 'ratio_kwh']

    Args:
        merge_how (str, optional): Can select "outer" or "inner". Defaults to "inner".

    Returns:
        _padas.Dataframe_: merged ami data.
    """
    result: DataFrame = open_mk33_may_haveratio(merge_how=merge_how)
    result.pop("ratio")
    result.pop("del_kwh")
    return result


@Timer(text="open mk33_6_7 file used time: {:0.4f} seconds", logger=logging.debug)
def open_mk33_6_7_file(merge_how: str = "inner") -> pd.DataFrame:
    """open and convert meter's files.
    Args:
        merge_how (str, optional): Can select "outer" or "inner". Defaults to "inner".

    Returns:
        _padas.Dataframe_: merged ami data.
    """
    mk33sl67_path: str = os.path.join(gl.get_value("INPUTPATH"), "mk33sl670601.csv")  # 設定抓取csv路徑
    # sl67_ratio_path: str = os.path.join(gl.get_value("INPUTPATH"), "sl67倍數.csv")  # 設定抓取csv路徑
    mk33sl67_data: DataFrame = pd.read_csv(mk33sl67_path)
    mk33_may: DataFrame = open_mk33_may_haveratio()
    mk33_may.drop(['cust_id', 'read_time', 'del_kwh', 'ratio_kwh'], axis=1, inplace=True)
    mk33_data: DataFrame = mk33sl67_data.set_index('饋線').loc['MK33'].reset_index(drop=True)
    mk33_ratio: DataFrame = (
        mk33_may.groupby("meter_id")['ratio']
        .apply(lambda x: list(np.unique(x)))
        .apply(lambda x: x[0])
        .reset_index()
    )
    result: DataFrame = pd.merge(mk33_data, mk33_ratio, on="meter_id", how=merge_how)
    result["ratio_kwh"] = result["ratio"] * result["del_kwh"]  # conculate ratio*kwh
    result.pop("del_kwh")
    result.pop("ratio")
    result.rename(columns={"電號": "cust_id"}, inplace=True)
    result = result[["cust_id", "meter_id", "read_time", "ratio_kwh"]]
    result["read_time"] = result["read_time"].map(
        {k: pd.to_datetime(k) for k in result["read_time"].unique()}
    )
    return result
