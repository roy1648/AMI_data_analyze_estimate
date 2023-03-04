#!/usr/bin/env python
# coding=utf-8
"""
File Description:
Author          : CHEN, JIA-LONG
Create Date     : 2023-02-11 13:50
FilePath        : \\N_20221213_ami_data_analyze\\N_20230211_main_estimate_transformer.py
Copyright © 2023 CHEN JIA-LONG.

將電表數據拉到全部，刪除太多和太少度數的用戶後，並改成7月或8月資料，
進行第一期期中報告的估測模型，對總用電量分群，找到該群裡面中心曲線最接近的用戶，
直接拿該用戶視為中心復原的根據，針對出來的資料，進行交叉驗證。
單相: B6744GD33_T01
三相: B6744GD85_T04
燈力：B6744GD15_T01
"""
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
    """Set the transformer groups to be calculated and the number of cross-validation iterations."""
    coordinate_group_three: list = ["B6744GD85", "T04"]
    coordinate_group_one: list = ["B6744GD33", "T01"]
    coordinate_group_uv: list = ["B6744GD15", "T01"]
    k_fold_rate: float = 0.2  # Set test rate: 0.2->20% test, 80% train.(Transformer)
    cross_validation_time: int = 5

    DEFINEFORMAT: str = "{:=^90s}"
    user_data: DataFrame = indata.open_user_detail()
    user_data.out.log(name="user_data", logger=logging.debug, savecsv=False)

    user_data_index: DataFrame = user_data.set_index(["圖號座標", "組別"])
    user_data_index.sort_index(inplace=True)
    three_phase: DataFrame = user_data_index.loc[
        (coordinate_group_three[0], coordinate_group_three[1])
    ].reset_index()
    one_phase: DataFrame = user_data_index.loc[
        (coordinate_group_one[0], coordinate_group_one[1])
    ].reset_index()
    uv_phase: DataFrame = user_data_index.loc[
        (coordinate_group_uv[0], coordinate_group_uv[1])
    ].reset_index()

    trans_data: DataFrame = indata.open_transformer_data()
    trans_data.out.log(name="transformer data", logger=logging.debug, savecsv=False)
    # data_july: DataFrame = indata.open_convert_file(merge_how="inner")  # May data
    data: DataFrame = indata.open_6_12_file()
    data_july: DataFrame = data.meter.choose_timerange(
        start_date="2022-07-01", end_date="2022-08-01"
    )  # remain about 430 user.
    data_july = data_july.sort_values(by=["cust_id", "meter_id", "read_time"])
    data_july.reset_index(drop=True, inplace=True)
    data_july = data_july.meter.fill_time_series_with_zeros()
    data_july.out.log(name="July", logger=logging.debug)

    pre_data_jul: DataFrame = data_july.meter.preprocess(logger=logging.warning)  # remain 365 user.
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

    """Calculate the correct utilization rate of the transformer."""
    correct_uti_rate_three_phase: DataFrame = out.load_uti_rate(
        coor=coordinate_group_three[0],
        group=coordinate_group_three[1],
        meter_data=data_july,  # no have 2976.
        trans_data=trans_data,
        user_data=user_data,
    )
    correct_uti_rate_one_phase: DataFrame = out.load_uti_rate(
        coor=coordinate_group_one[0],
        group=coordinate_group_one[1],
        meter_data=data_july,
        trans_data=trans_data,
        user_data=user_data,
    )
    correct_uti_rate_uv_phase: Tuple = out.load_uti_rate(
        coor=coordinate_group_uv[0],
        group=coordinate_group_uv[1],
        meter_data=data_july,
        trans_data=trans_data,
        user_data=user_data,
    )

    """Cross validation code."""
    accuracy_three_phase: list = []
    accuracy_one_phase: list = []
    accuracy_uv_phase_res: list = []
    accuracy_uv_phase_ind: list = []
    accuracy: list = []
    for count in range(1, cross_validation_time + 1):
        logging.info(DEFINEFORMAT.format(f"Start {count} time."))
        out.generate_root(f"{count}")

        """shuffle transformer data"""
        cust_three_train: DataFrame
        cust_three_test: DataFrame
        cust_three_train, cust_three_test = train_test_split(three_phase, test_size=k_fold_rate)
        cust_three_train.out.log(name="cust_three_train", logger=logging.debug)
        cust_three_test.out.log(name="cust_three_test", logger=logging.debug)

        cust_one_train: DataFrame
        cust_one_test: DataFrame
        cust_one_train, cust_one_test = train_test_split(one_phase, test_size=k_fold_rate)
        cust_one_train.out.log(name="cust_one_train", logger=logging.debug)
        cust_one_test.out.log(name="cust_one_test", logger=logging.debug)

        cust_uv_train: DataFrame
        cust_uv_test: DataFrame
        cust_uv_train, cust_uv_test = train_test_split(uv_phase, test_size=k_fold_rate)
        cust_uv_train.out.log(name="cust_uv_train", logger=logging.debug)
        cust_uv_test.out.log(name="cust_uv_test", logger=logging.debug)
        """Process tran and test data."""
        pre_data_jul_: DataFrame = copy.deepcopy(pre_data_jul)
        delete_data_: DataFrame = copy.deepcopy(delete_data)

        def del_cust() -> None:
            delete_data_.meter.del_specific_cust_data(cust_id=cust, inplace=True)
            if not pre_data_jul_.meter.get_specific_cust_kwh(cust=cust).empty:
                pre_data_jul_.meter.del_specific_cust_data(cust_id=cust, inplace=True)
            logging.debug(f"Deleted {cust}")
            # pre_data_jul_.meter.get_specific_cust_kwh(cust=cust).out.log(
            #     name=f"Need delete {cust}", logger=logging.debug
            # )

        test_cust_id: list = []
        data_july_index: DataFrame = data_july.set_index("cust_id")
        three_phase_data_error: bool = False
        one_phase_data_error: bool = False
        uv_phase_data_error: bool = False
        cust_test: list = [
            *list(cust_three_test["表號"]),
            *list(cust_one_test["表號"]),
            *list(cust_uv_test["表號"]),
        ]
        for cust in cust_test:
            try:
                test_cust_id.append(
                    list(data_july_index.loc[cust].groupby(["meter_id"]).groups.keys())[0]
                )
            except KeyError as e:
                if e.args[0] in list(cust_one_test["表號"]):
                    logging.error(f"No have cust_id 15min data(one phase): {e}")
                    one_phase_data_error = True
                elif e.args[0] in list(cust_three_test["表號"]):
                    logging.error(f"No have cust_id 15min data(three phase): {e}")
                    three_phase_data_error = True
                elif e.args[0] in list(cust_uv_test["表號"]):
                    logging.error(f"No have cust_id 15min data(uv phase): {e}")
                    uv_phase_data_error = True
                    # break  # with continue.
                else:
                    raise KeyError(e)
            del_cust()
        if one_phase_data_error and three_phase_data_error and uv_phase_data_error:
            logging.warning(
                "Since the data cannot be aligned this time,"
                + f" resampling is necessary for a new estimation:{count}"
            )
            continue
        # if uv_phase_data_error:
        #     logging.warning(f"continue for{count}")
        #     continue

        y_test: DataFrame = data_july.set_index("meter_id").loc[test_cust_id].reset_index()
        x_test: Series = y_test.meter.summary_all_meter
        """Build k-means estimate module"""
        sum_data: ndarray
        all_meter_data: DataFrame
        sum_data, all_meter_data = pre_data_jul_.meter.kmeans_preprocess_total
        eblow: int = km.SSE(cluster_max=20, data=sum_data)
        km_module: KMeans = km.kmeans(
            n_clusters=eblow,
            summary_all_meter=all_meter_data.meter.summary_all_meter,
            data=sum_data,
            logger=logging.debug,
        )
        km.draw_km_result_summary(data=pre_data_jul_, km_module=km_module)
        estimate_module: DataFrame = km.estimate_module(
            data=pre_data_jul_, km_module=km_module, draw=True
        )  # train estimate module
        estimate_module.out.log(name="estimate", logger=logging.debug)
        y_test_estimate: DataFrame = km.estimate(
            x_test=x_test,
            y_train=pre_data_jul_,
            km_module=km_module,
            estimate_module=estimate_module,
        )  # do traditional meter estimate.

        """Check estimate module and calculate meter estimate accuracy(Euclidean distance)."""
        km.estimate_check_currect(
            correct=y_test, estimate=y_test_estimate
        )  # Check estimate module is right.
        acc: float = km.estimate_accurate(
            correct=y_test, estimate=y_test_estimate, draw=True
        )  # y_test no have 2976.
        accuracy.append(acc)
        logging.info(f"Meters Euclidean distance average = {acc}")
        print(f"Meters Euclidean distance average = {acc}")

        """Fill custom id to estimate result data."""
        y_test_estimate.set_index(["meter_id"], inplace=True)
        y_test_fill0: DataFrame = (
            y_test.set_index(["read_time"])
            .groupby("meter_id")
            .resample("15min")
            .sum()
            .reset_index()
        )  # Resample data on the time axis every 15 minutes and fill in missing values with 0.
        time: list[pd.Timestamp] = list(y_test_estimate.groupby(["read_time"]).groups.keys())
        time.sort()
        empty_data: DataFrame = pd.DataFrame(0, index=pd.Index(time), columns=["ratio_kwh"])
        empty_data.index.name = "read_time"

        for meter in y_test_estimate.groupby(["meter_id"]).groups.keys():
            cust_id: int = list(
                y_test.set_index("meter_id").loc[meter].groupby("cust_id").groups.keys()
            )[0]
            y_test_estimate.loc[meter, "cust_id"] = [str(cust_id)] * y_test_estimate.loc[
                meter, "read_time"
            ].size

        y_test_estimate.reset_index(inplace=True)
        y_test_estimate.out.log(name="y_test_estimate", logger=logging.debug)

        all_meter_estimate: DataFrame = pd.concat(
            [y_test_estimate, pre_data_jul_, delete_data_]
        )  # Merge to all meter data.
        all_meter_estimate.out.log(name="all_meter_estimate", logger=logging.debug)

        """Calculate Transformer's utilization rate and accuracy(MAPE)."""
        if not three_phase_data_error:
            esti_uti_rate_three_phase: DataFrame = out.load_uti_rate(
                coor=coordinate_group_three[0],
                group=coordinate_group_three[1],
                meter_data=all_meter_estimate,
                trans_data=trans_data,
                user_data=user_data,
            )

            acc_uti_three: float = km.estimate_acc_trans_uti_rate(
                coor_group=f"{coordinate_group_three}",
                correct=correct_uti_rate_three_phase,
                estimate=esti_uti_rate_three_phase,
                draw=True,
            )
            logging.debug(f"Three phase transformer accuracy({count}) = {acc_uti_three}")
            accuracy_three_phase.append(acc_uti_three)

        if not one_phase_data_error:
            esti_uti_rate_one_phase: DataFrame = out.load_uti_rate(
                coor=coordinate_group_one[0],
                group=coordinate_group_one[1],
                meter_data=all_meter_estimate,
                trans_data=trans_data,
                user_data=user_data,
            )

            acc_uti_one: float = km.estimate_acc_trans_uti_rate(
                coor_group=f"{coordinate_group_one}",
                correct=correct_uti_rate_one_phase,
                estimate=esti_uti_rate_one_phase,
                draw=True,
            )
            logging.debug(f"One phase transformer accuracy({count}) = {acc_uti_one}")
            accuracy_one_phase.append(acc_uti_one)

        if not uv_phase_data_error:
            esti_uti_rate_uv_phase: Tuple = out.load_uti_rate(
                coor=coordinate_group_uv[0],
                group=coordinate_group_uv[1],
                meter_data=all_meter_estimate,
                trans_data=trans_data,
                user_data=user_data,
            )
            acc_uti_uv: float = km.estimate_acc_trans_uti_rate(
                coor_group=f"{coordinate_group_uv}",
                correct=correct_uti_rate_uv_phase,
                estimate=esti_uti_rate_uv_phase,
                draw=True,
            )
            logging.debug(f"resident_industry phase transformer accuracy({count}) = {acc_uti_uv}")
            accuracy_uv_phase_res.append(acc_uti_uv[0])
            accuracy_uv_phase_ind.append(acc_uti_uv[1])

    """Show the result of each time."""
    if len(accuracy_three_phase) != 0:
        logging.info(
            f"Every time accuracy_three phase = {len(accuracy_three_phase)}, {accuracy_three_phase}"
        )
        logging.info(
            f"Accuracy three phase mean({len(accuracy_three_phase)}) = "
            + f"{sum(accuracy_three_phase)/len(accuracy_three_phase)}"
        )
    else:
        logging.warning("No run to three phase transformer.")
    if len(accuracy_one_phase) != 0:
        logging.info(
            f"Every time accuracy_one phase = {len(accuracy_one_phase)}, {accuracy_one_phase}"
        )
        logging.info(
            f"Accuracy one phase mean({len(accuracy_one_phase)}) = "
            + f"{sum(accuracy_one_phase)/len(accuracy_one_phase)}"
        )
    else:
        logging.warning("No run to one phase transformer.")
    if len(accuracy_uv_phase_res) != 0:
        logging.info(
            f"Every time accuracy uv phase(resident)="
            + f"{len(accuracy_uv_phase_res)}, {accuracy_uv_phase_res}"
        )
        logging.info(
            f"Accuracy uv phase mean({len(accuracy_uv_phase_res)}) = "
            + f"{sum(accuracy_uv_phase_res)/len(accuracy_uv_phase_res)}"
        )
    else:
        logging.warning("No run to uv phase transformer(resident).")
    if len(accuracy_uv_phase_ind) != 0:
        logging.info(
            f"Every time accuracy uv phase(industry)="
            + f"{len(accuracy_uv_phase_ind)},{accuracy_uv_phase_ind}"
        )
        logging.info(
            f"Accuracy uv phase mean({len(accuracy_uv_phase_ind)}) = "
            + f"{sum(accuracy_uv_phase_ind)/len(accuracy_uv_phase_ind)}"
        )
    else:
        logging.warning("No run to uv phase transformer(industry).")
    logging.info(f"Every time accuracy_meter = {accuracy}")
    logging.info(f"Accuracy meter mean = {sum(accuracy)/len(accuracy)}")


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
            os.path.join(gl.get_value("OUTPUTLOGPATH"), "estimate.log"),
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
