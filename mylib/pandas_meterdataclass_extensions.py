#!/usr/bin/env python
# coding=utf-8
"""
File Description: If you need to typing hints, you need to use the below code in the main code.
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        class DataFrame(pd.DataFrame):
            meter: MeterDataAccessor
        class Series(pd.Series):
            meter: MeterDataAccessor

Author          : CHEN, JIA-LONG
Create Date     : 2022-12-30 20:46
FilePath        : \\N_20221213_ami_data_analyze\\N_20221116_MeterDataClass.py
Copyright © 2023 CHEN JIA-LONG.
"""

import copy
import logging
import re
from datetime import datetime
from io import StringIO
from typing import TYPE_CHECKING, Callable, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas.core.groupby.generic import DataFrameGroupBy
from pandas.core.indexes.api import DatetimeIndex
from pandas.core.indexes.base import Index
from sklearn.model_selection import train_test_split

from . import globalvar as gl
from . import module_OutputData as out
from .timer import Timer


@pd.api.extensions.register_series_accessor("out")
@pd.api.extensions.register_dataframe_accessor("out")
class OutAccessor:
    """Programming of the output variable."""

    def __init__(self, pandas_obj: Union[pd.DataFrame, pd.Series]) -> None:
        """When first call module, will initial setting.

        Args:
            pandas_obj (pandas_obj): call itself variable.
        """
        self._obj: Union[pd.DataFrame, pd.Series] = pandas_obj

    def log(
        self,
        name: Optional[str] = None,
        logger: Callable[[str], None] = logging.info,
        savecsv: bool = False,
    ) -> None:
        """Log specified Data in the Logfile.

        Args:
            name (string | None, optional): Data name. Defaults to None.
            logger ((str)-> None), optional): Can swich logging.info, logging.debug, print.
                                              Defaults to logging.info.
            savecsv (bool, optional): If True, will save csv file. Defaults to False.
        """
        defineformat: str = "{:=^90s}"
        if self._obj._typ == "dataframe":
            _buf: StringIO = StringIO()
            self._obj.info(buf=_buf)
            if name is None:
                logger(defineformat.format("DataFrame Data"))
            else:
                logger(defineformat.format(f"{name} Data"))
            logger(_buf.getvalue())
            logger(f"\n{self._obj}")
        elif self._obj._typ == "series":
            if name is None:
                logger(defineformat.format("Series Data"))
            else:
                logger(defineformat.format(f"{name} Series Data"))
            logger(self._obj)
        if savecsv is True:
            out.out_csv(datas=self._obj, filename=name, index=True, logger=logger)


@pd.api.extensions.register_series_accessor("meter")
@pd.api.extensions.register_dataframe_accessor("meter")
class MeterDataAccessor(OutAccessor):
    """Programming of treat meter variable."""

    def __init__(self, pandas_obj: Union[pd.DataFrame, pd.Series]) -> None:
        """When first call module, will initial setting.

        Args:
            pandas_obj (pandas_obj): call itself variable.
        """
        self._validate(pandas_obj)
        self._obj: Union[pd.DataFrame, pd.Series, DataFrame, Series] = pandas_obj

    @staticmethod
    def _validate(obj: pd.DataFrame) -> None:
        """verify are columns for date, tmin, tmax, and tavg

        Args:
            obj (pandas_obj): itself variable.

        Raises:
            AttributeError: Variable column have
                            '圖號座標','組別','cust_id','meter_id','read_time', 'ratio_kwh'
        """
        if obj._typ != "series":
            # col: list[str] = ["圖號座標", "組別", "cust_id", "meter_id", "read_time", "ratio_kwh"]
            col: list[str] = ["meter_id", "read_time", "ratio_kwh"]
            if not set(col).issubset(obj.columns):
                raise AttributeError(f"Columns must include {col}.")

    @property
    def summary_all_meter(self) -> pd.Series:
        """summary all meter's ratio_kwh from data

        Returns:
            _pandas_sereis_: summary all meter's ratio_kwh from data
        """
        return self._obj.groupby(["meter_id"])["ratio_kwh"].sum().sort_values()

    @property
    @Timer(text="normalize time: {:0.4f} seconds", logger=None)
    def normalize(self) -> pd.DataFrame:  # 確定是不是每次呼叫都會更新normalize?
        """Normalize Meter_kwh value.

        Returns:
            pd.DataFrame: Dataframe with added normalized value in columns.
        """
        _temp: DataFrame = self._obj.copy(deep=True)
        _sumaary: Series = self._obj.groupby(["meter_id"])["ratio_kwh"].sum()
        _datagroup: DataFrameGroupBy = _temp.groupby(["meter_id"])
        for _i in _datagroup.indices:
            _temp.loc[_datagroup.groups[_i], ["normalize"]] = (
                _temp["ratio_kwh"][_datagroup.groups[_i]] * 1000 / _sumaary[_i]
            )
        return _temp

    @property
    def weekday(self) -> "pd.Series[int]":
        """
        pd.Series[int]: Return to the day of week according to read_time.
                        1:MON, 2:TUE, 3:WED, 4:THU, 5:FRI, 6:SAT, 0:SUN.
        """
        return self._obj.read_time.dt.dayofweek

    @property
    def workingday_or_holiday(self) -> pd.DataFrame:
        """pd.DataFrame: Dataframe with added holiday and workingday value in columns."""
        holiday: Series[bool] = self.weekday > 4
        workingday: Series[bool] = self.weekday < 5
        _result: pd.DataFrame = copy.deepcopy(self._obj)
        _result['holiday'] = _result['ratio_kwh'][holiday]
        _result['workingday'] = self._obj['ratio_kwh'][workingday]
        return _result

    @property
    def kmeans_preprocess_total(self) -> Tuple[ndarray, pd.DataFrame]:
        """Preprocess data to use kmeans.

        Returns:
            Tuple[ndarray, pd.DataFrame]:
                ndarray: Total electricity number converts into ndarray.
                pd.DataFrame: Dataframe with added week value in columns.
        """
        kmeans_preprocess_summary: ndarray = self.summary_all_meter.values.reshape(
            -1, 1
        )  # 不能是一維矩陣，要轉成二維矩陣
        _temp: DataFrame = copy.deepcopy(self._obj)
        _temp['第幾周'] = _temp['read_time'].dt.isocalendar()['week']  # 要多一個.dt才能一次全部轉換，只取第幾周的欄位
        return kmeans_preprocess_summary, _temp

    @property
    def preprocess_1HR(self) -> pd.DataFrame:
        """Conversion time format.(month-day-hour)

        Returns:
            pd.DataFrame: Dataframe with added 'mm-dd:H' in columns.
        """
        _result: DataFrame = copy.deepcopy(self._obj)
        _result['mm-dd:H'] = _result['read_time'].dt.strftime('%m-%d:%H')  # 提取read_time的月份和日期和小時
        return _result

    @property
    def read_time_1hr(self) -> "pd.Series[str]":
        """Conversion time format.(month-day-hour)

        Returns:
            pd.Series[str]: Return convered time format.
        """
        return self._obj['read_time'].dt.strftime('%m-%d:%H')  # 提取read_time的月份和日期和小時

    @property
    def read_time_HMS(self) -> "pd.Series[str]":
        """
        Returns:
            pd.Series[str]: Extract the hour, minute and second time part of Read_time,
                            and convert it into a string.
        """
        return self._obj['read_time'].dt.strftime('%H:%M:%S')

    @property
    def week(self) -> "pd.Series[int]":
        """pd.Series[int]: Return the week according to read_time."""
        return self._obj['read_time'].dt.isocalendar()['week']  # 要多一個.dt才能一次全部轉換，只取第幾周的欄位

    @property
    def night(self) -> pd.DataFrame:
        """pd.DataFrame: Return night time(23:00 -> 07:00) data."""
        night_range: DatetimeIndex = pd.date_range(
            start='2022-05-01 23:00:00', end='2022-05-02 07:00:00', freq='15min'
        )  # 創造特定時間區段_night
        str_HMS: Index = night_range.strftime('%H:%M:%S')  # 只取HMS時間的部分
        night_bool: Series[bool] = self.read_time_HMS.isin(list(str_HMS))
        return self._obj[night_bool]

    @property
    def daytime(self) -> pd.DataFrame:
        """pd.DataFrame: Return daytime(07:00 -> 23:00) data."""
        night_range: DatetimeIndex = pd.date_range(
            start='2022-05-01 23:00:00', end='2022-05-02 07:00:00', freq='15min'
        )  # 創造特定時間區段_night
        str_HMS: Index = night_range.strftime('%H:%M:%S')  # 只取HMS時間的部分
        night_bool: Series[bool] = self.read_time_HMS.isin(list(str_HMS))
        return self._obj[~night_bool]

    def complete_data_week(self, one_week_size: int) -> list:
        """Return have complete weekly vaule's week.

        Args:
            one_week_size (int): How much data is needed a week.

        Returns:
            list: Materials with a complete weekly value.
        """
        week: DataFrameGroupBy = self._obj.groupby(self.week)
        result: list = []
        for z in week.indices:
            if week.groups[z].size == one_week_size:
                result.append(z)
        return result

    def _create_week_dataframe(self, mode: str, time: str) -> pd.DataFrame:
        """Create week data needs empty dataframe.

        Args:
            mode (string): swich to create dataframe.


        Raises:
            NoModeError: If no this mode, will raise this error.

        Returns:
            _type_: mutiple index and columns dataframe.
        """
        meter_id: DataFrameGroupBy = self._obj.groupby(["meter_id"])
        index_lavel1: list = list(meter_id.groups)
        if mode == "night":
            data_type: DataFrame = copy.deepcopy(self.night)
        elif mode == "daytime":
            data_type: DataFrame = copy.deepcopy(self.daytime)
        elif mode == "alltime":
            data_type = copy.deepcopy(self._obj)
        else:
            raise ValueError(f"No have this mode:{mode}")
        index_lavel2: list = self.complete_data_week(one_week_size=96 * 7 * len(meter_id))
        index_iterables: list = [index_lavel1, index_lavel2]
        index: pd.MultiIndex = pd.MultiIndex.from_product(
            index_iterables, names=["meter_id", "week"]
        )
        if time == "15min":
            time_type: Series[str] = data_type['read_time'].dt.strftime('%w_%H:%M')
            match_str: str = "0_\d\d:\d\d"
        elif time == "1hr":
            time_type: Series[str] = data_type['read_time'].dt.strftime('%w_%Hhr')
            match_str: str = "0_\d\dhr"
        else:
            raise ValueError(f"No have this time: {time}")
        time_column_group: list = sorted(list(time_type.groupby(time_type).groups))
        i_l: list = []
        for i, v in enumerate(time_column_group):
            match: Optional[re.Match] = re.match(match_str, v)
            if bool(match):
                i_l.append(i)
        time_column: list[str] = (
            time_column_group[i_l[-1] + 1 :] + time_column_group[i_l[0] : i_l[-1] + 1]
        )
        return pd.DataFrame(index=index, columns=time_column)

    @property
    def week_night(self) -> pd.DataFrame:
        """Return night time that Monday to Sunday.Group to week.

        Returns:
            pd.DataFrame: _description_
                Index:
                    meter_id: string
                    week: int
                Columns:
                    Night time that Monday to Sunday. Each hour of data.
        """
        result = self._create_week_dataframe(mode="night", time="1hr")
        idx = pd.IndexSlice
        week: DataFrameGroupBy = self._obj.groupby(self.week)
        meter_id: DataFrameGroupBy = self._obj.groupby(["meter_id"])
        one_week_size: int = 96 * 7 * len(meter_id)
        night_time: DataFrame = copy.deepcopy(self.night)
        night_time: DataFrame = copy.deepcopy(self.night)
        for j in week.indices:
            if week.groups[j].size == one_week_size:
                night_time_a_meter: DataFrame = night_time[
                    night_time['read_time'].dt.isocalendar()['week'] == j
                ]
                night_time_a_meter.insert(
                    len(night_time_a_meter.columns), "1hr", night_time_a_meter.meter.read_time_1hr
                )
                night_ami_1hr: DataFrame = night_time_a_meter.groupby(['meter_id', '1hr']).agg(
                    {'ratio_kwh': 'sum'}
                )
                result.loc[idx[:, j], :] = night_ami_1hr.stack().unstack(level=1).values
        return result

    @property
    def week_value(self) -> pd.DataFrame:
        """Return time that Monday to Sunday.Group to week.

        Returns:
            pd.DataFrame: _description_
                Index:
                    meter_id: string
                    week: int
                Columns:
                    All time that Monday to Sunday. Each hour of data.
        """
        result = self._create_week_dataframe(mode="alltime", time="1hr")
        idx = pd.IndexSlice
        week: DataFrameGroupBy = self._obj.groupby(self.week)
        meter_id: DataFrameGroupBy = self._obj.groupby(["meter_id"])
        one_week_size: int = 96 * 7 * len(meter_id)
        for j in week.indices:
            if week.groups[j].size == one_week_size:
                week_meter: DataFrame = self._obj[
                    self._obj['read_time'].dt.isocalendar()['week'] == j
                ]
                week_meter.insert(len(week_meter.columns), "1hr", week_meter.meter.read_time_1hr)
                night_ami_1hr: DataFrame = week_meter.groupby(['meter_id', '1hr']).agg(
                    {'ratio_kwh': 'sum'}
                )
                result.loc[idx[:, j], :] = night_ami_1hr.stack().unstack(level=1).values
        return result

    @property
    def week_value_15min(self) -> pd.DataFrame:
        """Return time that Monday to Sunday.Group to week.

        Returns:
            pd.DataFrame: _description_
                Index:
                    meter_id: string
                    week: int
                Columns:
                    All time that Monday to Sunday. Each 15min of data.
        """
        meter_id: DataFrameGroupBy = self._obj.groupby(["meter_id"])
        result: DataFrame = copy.deepcopy(self._obj)
        result.insert(len(result.columns), "week", result['read_time'].dt.isocalendar()['week'])
        result.insert(
            len(result.columns), "modify_time", result['read_time'].dt.strftime('%w_%H:%M')
        )
        result.set_index(["meter_id", "week", "modify_time"], inplace=True)
        result.drop(axis=1, columns=["read_time", "圖號座標", "組別", "cust_id"], inplace=True)
        result = result.stack().unstack(level=2)
        result = result.droplevel(level=2)
        result.columns.name = None
        complete_week: list = self.complete_data_week(one_week_size=96 * 7 * len(meter_id))
        all_week: list = list(self._obj.groupby(self.week).groups.keys())
        different_week: list = list(set(all_week) - set(complete_week))
        result.drop(index=different_week, level="week", inplace=True)
        result.sort_index(inplace=True)
        i_l: list = []
        result_columns: list = list(result.columns)
        match_str: str = "0_\d\d:\d\d"
        for i, v in enumerate(result.columns):
            match: Optional[re.Match] = re.match(match_str, v)
            if bool(match):
                i_l.append(i)
        result_columns = result_columns[i_l[-1] + 1 :] + result_columns[i_l[0] : i_l[-1] + 1]
        result = result.reindex(result_columns, axis=1)
        return result

    def delete_over_value(self, val: int) -> pd.DataFrame:
        # Delete the value above the specified value.
        summary: Series = self.summary_all_meter
        bool = summary > val
        _result: DataFrame = copy.deepcopy(self._obj)
        _result.set_index("meter_id", inplace=True)
        _result.drop(index=summary[bool].index, inplace=True)
        _result.reset_index(inplace=True)
        cols: list[str] = list(_result.columns)
        cols = [cols[1], cols[0], *cols[2:]]
        _result = _result[cols]
        return _result

    def delete_below_value(self, val: int) -> pd.DataFrame:
        # Delete the value below the specified value.
        summary: Series = self.summary_all_meter
        bool = summary < val
        _result: DataFrame = copy.deepcopy(self._obj)
        _result.set_index("meter_id", inplace=True)
        _result.drop(index=summary[bool].index, inplace=True)
        _result.reset_index(inplace=True)
        cols: list[str] = list(_result.columns)
        cols = [cols[1], cols[0], *cols[2:]]
        _result = _result[cols]
        return _result

    def preprocess(
        self,
        quantity: int = 2976,
        logger: Optional[Callable[[str], None]] = logging.debug,
    ) -> pd.DataFrame:
        """Preprocess input data.

        Args:
            quantity (int, optional): Complete data number. Defaults to 2976.
            logger ((str)-> None) | None, optional): Can swich logging.info, logging.debug, print.
                                                     Defaults to logging.debug.

        Returns:
            pd.DataFrame: Have complete vaule and time data.
        """
        _result: DataFrame = self.delete_all_zero_data(logger=logger)
        _result = _result.meter.check_all_meter_timedata(logger=logger, quantity=quantity)
        return _result

    def col_duplicated(self, param_col: list = ['meter_id', 'read_time']) -> pd.DataFrame:
        """check have duplicated ami data?

        Args:
            param_col (list): To check the column with duplicate value .
                              Defaults to ['meter_id', 'read_time'].

        Returns:
            pd.DataFrame: Return the duplicate value.
        """
        return self._obj[self._obj.duplicated(subset=param_col, keep=False)]

    def delete_all_zero_data(
        self, logger: Optional[Callable[[str], None]] = logging.debug
    ) -> pd.DataFrame:
        """Delete meter with power is 0.
           Original data not be changed.

        Returns:
            pandas.Dataframe: If Meter_kwh is all zero, Meter_kwh is deleted.
        """
        _data_summary: pd.Series = self.summary_all_meter
        _sumamary_zero_meter: pd.Series = _data_summary[_data_summary.values == 0]
        if logger:
            logger(f"meter data is all zero:{_sumamary_zero_meter.index.values}")
        # logging.debug(_sumamary_zero_meter.index.values)
        _result: MeterDataAccessor = copy.deepcopy(self)
        # _result: pd.DataFrame = self._obj.copy(deep=True)
        for i in _sumamary_zero_meter.index:
            _temp = _result._obj[_result._obj.meter_id == i].index
            # logging.debug(self._obj[self._obj.meter_id == i].head(1))
            if logger:
                logger(f"{i} all zero data size: {_temp.size}")
            _result._obj.drop(_temp, inplace=True)
        return _result._obj

    def del_specific_data(self, meter_id: str, inplace: bool = True) -> Union[pd.DataFrame, None]:
        """Delete specific meter data. You can use fuction to delete strange data.

        Args:
            inplace (bool, optional): If True, do operation inplace and return None.
                                      Defaults to True.
            meter_id (str): Need to delete meter_id data.

        Returns:
            pd.DataFrame: Deleted data.
        """
        if inplace is True:
            _temp = self._obj[self._obj.meter_id == meter_id].index
            self._obj.drop(_temp, inplace=True)
        else:
            _temp = self._obj[self._obj.meter_id == meter_id].index
            _result: MeterDataAccessor = copy.deepcopy(self)
            _result._obj.drop(_temp, inplace=True)
            return _result._obj

    def del_specific_cust_data(
        self, cust_id: str, inplace: bool = True
    ) -> Union[pd.DataFrame, None]:
        """Delete specific meter data. You can use fuction to delete strange data.

        Args:
            inplace (bool, optional): If True, do operation inplace and return None.
                                      Defaults to True.
            cust_id (str): Need to delete cust_id data.

        Returns:
            pd.DataFrame: Deleted data.
        """
        if inplace is True:
            _temp = self._obj[self._obj.cust_id == int(cust_id)].index
            self._obj.drop(_temp, inplace=True)
            _temp = self._obj[self._obj.cust_id == str(cust_id)].index
            self._obj.drop(_temp, inplace=True)
        else:
            _temp = self._obj[self._obj.cust_id == int(cust_id)].index
            _result: MeterDataAccessor = copy.deepcopy(self)
            _result._obj.drop(_temp, inplace=True)
            _temp = self._obj[self._obj.cust_id == str(cust_id)].index
            _result._obj.drop(_temp, inplace=True)
            return _result._obj

    def check_all_meter_timedata(
        self, logger: Optional[Callable[[str], None]] = logging.debug, quantity: int = 2976
    ) -> pd.DataFrame:
        """Reserve meter data with complete time.

        Args:
            logger (str): Can swich logging.info, logging.debug, print. Defaults to logging.info.
            quantity (str): One month have value(31*24*4). Defaults to 2976.

        Returns:
            pd.DataFrame: Delete no complete value and return it back.
        """
        _temp: DataFrameGroupBy = self._obj.groupby(["meter_id"])

        _result = copy.deepcopy(self)

        for i in _temp.meter_id.indices:
            _j: int = _temp.meter_id.indices[i].size
            if _j != quantity:
                if logger:
                    logger(f"{i} need data size {quantity}, no have all data is:{_j}")
                _meterindex: pd.Index = _result._obj[_result._obj.meter_id == i].index
                _result._obj.drop(_meterindex, inplace=True)
        return _result._obj

    def get_specific_meter_kwh(self, meter: str) -> pd.DataFrame:
        """Get specific meter's data.

        Args:
            meter (Optional[str]): Specifc meter.

        Returns:
            pd.DataFrame: Return specific meter's data
        """
        return self._obj.loc[self._obj["meter_id"] == meter]

    def get_specific_cust_kwh(self, cust: str) -> pd.DataFrame:
        """Get specific cust's data.

        Args:
            cust (Optional[str]): Specifc cust.

        Returns:
            pd.DataFrame: Return specific cust's data
        """
        return self._obj.loc[self._obj["cust_id"].astype("int") == int(cust)]

    def choose_timerange(self, start_date: str, end_date: str) -> pd.DataFrame:
        mask: list[bool] = (self._obj["read_time"] >= start_date) & (
            self._obj["read_time"] < end_date
        )
        return self._obj.loc[mask]

    def shuffle_split(
        self, test_size: float, random: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        x_train: DataFrame
        x_test: DataFrame
        if random:
            x_train, x_test = train_test_split(
                self._obj.meter.summary_all_meter, test_size=test_size
            )
        else:
            x_train, x_test = train_test_split(
                self._obj.meter.summary_all_meter, test_size=test_size, random_state=1
            )
        data: DataFrame = copy.deepcopy(self._obj)
        data.set_index("meter_id", inplace=True)
        y_test: DataFrame = data.loc[x_test.index].reset_index(inplace=False)
        y_train: DataFrame = data.loc[x_train.index].reset_index(inplace=False)
        return x_train, x_test, y_train, y_test

    def fill_time_series_with_zeros(
        self, logger: Optional[Callable[[str], None]] = logging.warning
    ) -> pd.DataFrame:
        time: list = list(self._obj.groupby(["read_time"]).groups.keys())
        time.sort()
        dates: DatetimeIndex = pd.date_range(time[0], time[-1], freq='15min')

        idx: pd.MultiIndex = pd.MultiIndex.from_product(
            [self._obj["meter_id"].unique(), dates],
            names=['meter_id', 'read_time'],
        )
        _temp: DataFrameGroupBy = self._obj.groupby(["meter_id"])
        for i in _temp.meter_id.indices:
            _j: int = _temp.meter_id.indices[i].size
            if _j != dates.size:
                if logger:
                    logging.warning(
                        f"Meter {i} donesn't have complete time data({_j}),"
                        + f" which should have {dates.size} records. "
                        + "All missing data had been filled with 0."
                    )

        cust_ids: DataFrame = self._obj[["meter_id", "cust_id"]].drop_duplicates(subset="meter_id")
        result: DataFrame = self._obj.set_index(["meter_id", "read_time"]).reindex(idx).fillna(0)
        result.pop("cust_id")
        result = result.reset_index().merge(cust_ids, on="meter_id")
        return result
        # result: DataFrame = (
        #     self._obj.set_index(["meter_id", "read_time"]).reindex(idx).fillna(0).reset_index()
        # )
        # _index_self: DataFrame = self._obj.set_index(["meter_id"])
        # result.set_index(["meter_id"], inplace=True)

        # for i in _temp.meter_id.indices:
        #     cust: int = list(_index_self.loc[i].groupby("cust_id").groups.keys())[0]
        #     result.loc[i, "cust_id"] = [str(cust)] * len(time)
        # result.reset_index(inplace=True)


@pd.api.extensions.register_dataframe_accessor("error")
class ErrorAccessor(OutAccessor):
    def __init__(self, pandas_obj: pd.DataFrame) -> None:
        """When first call module, will initial setting.

        Args:
            pandas_obj (pandas_obj): call itself variable.
        """
        # self._validate(pandas_obj)
        self._obj: DataFrame = pandas_obj

    @property
    def mape(self) -> Tuple[pd.Series, pd.Series]:
        """Calculate MAPE(Compare every week).

        Returns:
            Tuple[pd.Series, pd.Series]:
                pd.Series: Weekly data comparison
                pd.Series: Weekly data comparison average
        """
        if set(['meter_id', 'week']).issubset(self._obj.index.names):
            idx = pd.IndexSlice
            week_index: list = list(self._obj.groupby(level=1).groups)
            meter_id_index = list(
                self._obj.groupby(self._obj.index.get_level_values('meter_id')).groups
            )
            index_lavel2: list = []
            for i in week_index[:-1]:
                i_index: int = week_index.index(i)
                for j in week_index[i_index + 1 :]:
                    index_lavel2.append(f'week_{i}{j}')

            index_iterables: list = [list(meter_id_index), index_lavel2]
            index: pd.MultiIndex = pd.MultiIndex.from_product(
                index_iterables, names=["meter_id", "week_compare"]
            )
            distance: DataFrame = pd.DataFrame(index=index, columns=self._obj.columns)

            def week_value(w) -> pd.DataFrame:
                return self._obj.loc[idx[:, w], :].astype('float').values

            np.seterr(divide='ignore', invalid='ignore')  # Set igonre week_value is 0 error.

            for i in week_index[:-1]:
                i_index: int = week_index.index(i)
                for j in week_index[i_index + 1 :]:
                    distance.loc[idx[:, f'week_{i}{j}'], :] = abs(
                        week_value(i) - week_value(j)
                    ) / week_value(i)
            distance.out.log(name="distance", logger=logging.debug, savecsv=True)
            similarity: Series = distance.sum(axis=1) * 100 / len(distance.columns)
            similarity.out.log(name="similarity", logger=logging.debug, savecsv=True)
            similarity_meter: Series = similarity.groupby(level='meter_id').mean()
            similarity_meter.out.log(name="similarity_meter", logger=logging.debug, savecsv=True)
            return similarity, similarity_meter
        else:
            _temp: DataFrame = self._obj.meter.week_night
            return _temp.error.mape


if TYPE_CHECKING:

    class DataFrame(pd.DataFrame):
        meter: MeterDataAccessor
        error: ErrorAccessor
        out: OutAccessor

    class Series(pd.Series):
        meter: MeterDataAccessor
        out: OutAccessor
