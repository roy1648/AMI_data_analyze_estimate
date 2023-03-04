#!/usr/bin/env python
# coding=utf-8
"""
File Description: Output module file.
Author          : CHEN, JIA-LONG
Create Date     : 2022-12-30 20:46
FilePath        : \\N_20221213_ami_data_analyze\\module_OutputData.py
Copyright © 2023 CHEN JIA-LONG.
"""

import copy
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

import matplotlib  # matplotlib.use('Agg') #如果要大量存圖在打開，打開時會使plt.show()失效，避免memory over

matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.pyplot import plot
from memory_profiler import profile
from numpy import array, ndarray
from pandas.core.arrays.boolean import BooleanArray
from pandas.core.groupby.generic import DataFrameGroupBy
from pandas.core.indexes.api import DatetimeIndex
from pandas.core.indexes.base import Index
from sklearn.cluster import KMeans

from . import globalvar as gl
from .timer import Timer

if TYPE_CHECKING:  # Using for visual studio code type hint.
    from .pandas_meterdataclass_extensions import ErrorAccessor, MeterDataAccessor, OutAccessor

    class DataFrame(pd.DataFrame):
        meter: MeterDataAccessor
        error: ErrorAccessor
        out: OutAccessor

    class Series(pd.Series):
        meter: MeterDataAccessor
        out: OutAccessor


@dataclass
class FigPara:
    """A class to save or show plot result.
       Set default Figure parament, you can set the suitable value.

    Attributes:
        xlabel (None | str, optional): X-axis label. Defaults to None.
        ylabel (None | str, optional): Y-axis label. Defaults to None.
        title (None| str, optional): file name and plot title. Defaults to None.
        generate_newfile (None | str, optional): If set, picture will save a same folder. Defaults to None.
        logger (None | (str)->None), optional): If set, program will recode in logfile. Defaults to None.
        figsize (list[int,int], optional): Figure size. Defaults to field(default_factory=lambda: [30, 20]).
        labelsize (int, optional): Axis label size. Defaults to 32.
        titlesize (int, optional): Title size. Defaults to 32.
        tickssize (int, optional): Tick size. Defaults to 25.
        ticks_rotation (int, optional): Ticks rotation. Defaults to 45.
        ticks_rotation_x (int, optional): X-Ticks rotation. Defaults to 0.
        ticks_rotation_y (int, optional): Y-Ticks rotation. Defaults to 0.
        legend_size (int, optional): Legned size. Defaults to 20.
        dpi (int, optional): Figure DPI. Defaults to 300.
        pad_inches (float, optional): Figure pad_inches. Defaults to 0.1.
        markersize (int, optional): Plot marksize. Defaults to 25.
        linewidth (float, optional): Plot line width. Defaults to 0.9.
        gird (bool, optional): If True, show figure grid. Defaults to True.
        grid_linestyle (str, optional): Figure grid's linestyle. Defaults to "--".
        bbox_inches (str, optional): Figure bbox inches. Defaults to "tight".
        show (bool, optional): If True, Figure will show. Defaults to False.
        save (bool, optional): If True, Figure will save. Defaults to True.
        __t (Timer, optional): Use to Timeing draw, can't be modify.
    """

    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    title: Optional[str] = None
    generate_newfile: Optional[str] = None
    logger: Optional[Callable[[str], None]] = None
    xlim: List[int] = None
    ylim: List[int] = None
    figsize: Optional[List[int]] = field(default_factory=lambda: [30, 20])
    labelsize: int = 32
    titlesize: int = 32
    tickssize: int = 25
    ticks_rotation: int = 45
    ticks_rotation_x: int = 0
    ticks_rotation_y: int = 0
    legend_size: int = 20
    dpi: int = 300
    pad_inches: float = 0.1
    markersize: int = 25
    linewidth: float = 0.9
    gird: bool = True
    grid_linestyle: str = "--"
    bbox_inches: str = "tight"
    show: bool = False
    save: bool = True
    __t: Timer = Timer(logger=None)

    def __enter__(self) -> "FigPara":
        """Create a plot figure and time start now.

        Returns:
            FigPara: "FigPara" class will return.
        """
        if self.logger is not None:
            self._FigPara__t.start()
        return self.openfig()

    def __exit__(self, *exc_info) -> None:
        """Save or show plot figure, release memory, end of timeing."""
        self.savefig()
        if self.logger is not None:
            _time: float = self._FigPara__t.stop()
            if self.title is not None:
                self.logger(f"Save {self.title} figure use {_time:0.2f} seconds.")
            else:
                self.logger(f"Save figure use {_time: 0.2f} seconds.")

    def openfig(self) -> "FigPara":
        """Create a plot Figure.

        Returns:
            FigPara: "FigPara" class will return.
        """
        if self.figsize:
            self.fig: Figure = plt.figure(figsize=self.figsize)
        else:
            self.fig: Figure = plt.figure()
        return self

    # @profile  # Record memory use status.
    def savefig(self) -> None:
        """Format plot window, save or show this window."""
        plt.grid(self.gird, linestyle=self.grid_linestyle)
        if self.xlabel is not None:
            plt.xlabel(self.xlabel, size=self.labelsize)
        if self.ylabel is not None:
            plt.ylabel(self.ylabel, size=self.labelsize)
        plt.xticks(rotation=self.ticks_rotation_x, size=self.tickssize)
        plt.yticks(rotation=self.ticks_rotation_y, size=self.tickssize)
        if self.xlim is not None:
            plt.xlim(self.xlim[0], self.xlim[1])
        if self.ylim is not None:
            plt.xlim(self.ylim[0], self.ylim[1])
        generate_outfile()
        if self.generate_newfile is not None:
            _draw_save_path: str = generate_newfile(self.generate_newfile)
        else:
            _draw_save_path: str = gl.get_value("OUTPUTPATH")
        if self.title is None:
            now: str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            path_name = os.path.join(_draw_save_path, f"temp_picture_{now}.jpg")
        else:
            plt.title(f"{self.title}", size=self.titlesize)
            path_name = os.path.join(_draw_save_path, f"{self.title}.jpg")
        if self.save:
            plt.savefig(
                path_name,
                dpi=self.dpi,
                bbox_inches=self.bbox_inches,
                pad_inches=self.pad_inches,
            )
        if self.show:
            plt.ion()
            plt.show()
            plt.pause(0.000000001)
        else:
            plt.clf()
            self.fig.clear()
            plt.close(self.fig)


def generate_outfile() -> None:
    """
    Be sure to do these things before execution:
        1)import globalvar.py file.
        2)"OUTPUTLOGPATH"(global_value) has been set.

    It's will create oupute file and set "OUTPUTPATH" in globalvar.py.
    OUTPUTPATH: Enviroment path\\output\\(prjectname)_out\\day\\time
    """
    if gl.check_value("OUTPUTPATH"):
        pass
    else:
        day: str = datetime.now().strftime("%Y%m%d")
        create_file: str = f"{day}"
        filename: str = gl.get_value("CODEFILE", defvalue="Temp")
        now: str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        time_file: str = f"{now}"
        path: str = os.path.join(
            gl.get_value("OUTPUTLOGPATH"), f"{filename}_out", create_file, time_file
        )
        os.makedirs(path, exist_ok=True)
        gl.set_value("OUTPUTPATH", path, readonly=False)


def generate_root(root: str, state: str = "generate") -> str:
    generate_outfile()
    if state == "generate":
        if gl.check_value("OUTPUTPATH_SAVE") is False:
            gl.set_value("OUTPUTPATH_SAVE", gl.get_value("OUTPUTPATH"), readonly=True)
            root_path: str = generate_newfile(filename=root)
            gl.set_value("OUTPUTPATH", root_path)
        else:
            gl.set_value("OUTPUTPATH", gl.get_value("OUTPUTPATH_SAVE"))
            root_path: str = generate_newfile(filename=root)
            gl.set_value("OUTPUTPATH", root_path)
    elif state == "return":
        if gl.check_value("OUTPUTPATH_SAVE") is False:
            gl.set_value("OUTPUTPATH", gl.get_value("OUTPUTPATH_SAVE"))
    else:
        raise ValueError(f"No have state:{state}")
    return gl.get_value("OUTPUTPATH")


def generate_newfile(filename: str) -> str:
    """Use function to create new file under output file path.

    Args:
        filename (str): New file name.

    Returns:
        str: New file's path.
    """
    generate_outfile()
    path: str = os.path.join(gl.get_value("OUTPUTPATH"), filename)
    if gl.check_value(f"OUTPUTPATH_{filename}"):
        if path == gl.get_value(f"OUTPUTPATH_{filename}"):
            return gl.get_value(f"OUTPUTPATH_{filename}")
        else:
            os.makedirs(path, exist_ok=True)
            gl.set_value(
                f"OUTPUTPATH_{filename}",
                path,
                readonly=False,
            )
            return path
    else:
        os.makedirs(path, exist_ok=True)
        gl.set_value(
            f"OUTPUTPATH_{filename}",
            path,
            readonly=False,
        )
        return path


# @Timer(text="Output csv file time:{:0.4f} seconds.", logger=logging.debug)
def out_csv(
    datas: Union[pd.DataFrame, pd.Series],
    sep: str = ",",
    filename: Optional[str] = None,
    index: bool = False,
    logger: Optional[Callable[[str], None]] = logging.debug,
) -> None:
    """Generate output file and output data csv file.

    Args:
        datas (pd.DataFrame, pd.Series): Need output data.
        sep (str, optional): Csv separate sign. Defaults to ",".
        filename (str, optional): Output file name. Defaults to None.
        index (bool, optional): Do you need output data indexes?. Defaults to False.
    """
    _usetime: Timer = Timer(logger=None)
    _usetime.start()
    generate_outfile()
    now: str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if filename is None:
        filename: str = f"out_temp_{now}"
    else:
        filename = f"{filename}_{now}"
    datas.to_csv(
        f"{gl.get_value('OUTPUTPATH')}\\{filename}.csv",
        sep=sep,
        na_rep="None",
        index=index,
        encoding="utf-8-sig",
    )  # 輸出csv檔案
    if logger is not None:
        logger(f"{filename}.csv has been established and used {_usetime.stop():0.4f} seconds.")


@Timer(text="draw fuction used time:{:0.4f} seconds", logger=logging.debug)
def all_meter_draw(
    datas_draw: pd.DataFrame,
    date: list = None,
    ylim: float = None,
    y: str = "ratio_kwh",
    xlabel: str = "time",
    ylabel: str = "kwh/15min",
    title: str = None,
) -> None:
    """draw all meter

    Args:
        datas_draw (pd.DataFrame): Data that need to be drawn.
        date (list, optional): The time range that needs to be drawn.[start_time, end_time]
                               Value need datetime format.
                               Defaults to None.
        ylim (float, optional): y axis draw limit. Defaults to None.
        y (str, optional): The data column needs to be drawn. Defaults to "ratio_kwh".
        xlabel (str, optional): xlabel's name. Defaults to "time".
        ylabel (str, optional): ylabel's name. Defaults to "kwh/15min".
        title (str, optional): Draw title name. Defaults to None.
    """
    fig_para: FigPara = FigPara()
    generate_outfile()
    fig: Figure = plt.figure(figsize=fig_para.figsize)
    datas_draw: pd.DataFrame = datas_draw.sort_values(
        by=["meter_id", "cust_id", "read_time"], ignore_index=True
    )
    datas_draw_class: DataFrameGroupBy = datas_draw.groupby("meter_id")
    datas_draw["day_of_week"] = datas_draw["read_time"].dt.dayofweek

    for index in datas_draw_class.indices:
        temp_val2: list[int] = datas_draw_class.indices[index]
        if date is None:
            temp_val: array = array(datas_draw.read_time[temp_val2])
            temp_val1: array = array(datas_draw[y][temp_val2])
        else:
            start_date: str = date[0]
            end_date: str = date[1]
            mask: list[bool] = (datas_draw["read_time"] > start_date) & (
                datas_draw["read_time"] <= end_date
            )
            temp_val = array(datas_draw.read_time[temp_val2].loc[mask])
            temp_val1 = array(datas_draw[y][temp_val2].loc[mask])
        plot(temp_val, temp_val1, linewidth=fig_para.linewidth)
    plt.xlabel(xlabel, size=fig_para.labelsize)
    plt.ylabel(ylabel, size=fig_para.labelsize)
    plt.grid(visible=fig_para.gird)
    if ylim is not None:
        plt.ylim(0, ylim)
    plt.title(title, size=fig_para.titlesize)
    if date is None:
        plt.savefig(
            f"{gl.get_value('OUTPUTPATH')}{title}.jpg",
            dpi=1000,
            bbox_inches="tight",
            pad_inches=fig_para.pad_inches,
        )
    else:
        plt.savefig(
            f"{gl.get_value('OUTPUTPATH')}{title}{date[0]}~{date[1]}.jpg",
            dpi=1000,
            bbox_inches="tight",
            pad_inches=fig_para.pad_inches,
        )
    fig.clear()
    plt.close(fig)


@Timer(text="draw_onegroup_plot used time: {:0.4f} seconds")
def draw_onegroup_plot(
    datas: pd.DataFrame,
    y: str = "ratio_kwh",
    xlabel: str = "time",
    ylabel: str = "kwh/15min",
    subtitle: str = None,
) -> None:
    """draw group transformer's meter plot

    Args:
        datas (pd.DataFrame): Data that need to be drawn.
        y (str, optional): The data column needs to be drawn. Defaults to "ratio_kwh".
        xlabel (str, optional): xlabel's name. Defaults to "time".
        ylabel (str, optional): ylabel's name. Defaults to "kwh/15min".
        subtitle (str, optional): Draw subtitle name. Defaults to None.
    """
    fig_para: FigPara = FigPara()
    generate_outfile()
    datas: pd.DataFrame = datas.sort_values(
        by=["圖號座標", "組別", "cust_id", "meter_id", "read_time"], ignore_index=True
    )
    datas_user_class: DataFrameGroupBy = datas.groupby(["圖號座標", "組別"])
    fig: Figure = plt.figure(figsize=fig_para.figsize)
    for i in datas_user_class.indices:
        meter: pd.DataFrame = datas_user_class[["meter_id"]].get_group(
            (i[0], i[1])
        )  # 選取在這個分群下所有的電表編號
        temp_class: DataFrameGroupBy = meter.groupby(["meter_id"])
        for j_val in temp_class.indices:
            c: pd.Series[bool] = datas["meter_id"] == j_val
            a: ndarray = datas["read_time"].to_numpy()[c]
            b: ndarray = datas[y].to_numpy()[c]
            plt.plot(a, b, linewidth=fig_para.linewidth)
        plt.grid(True)
        plt.xlabel(xlabel, size=fig_para.labelsize)
        plt.ylabel(ylabel, size=fig_para.labelsize)
        plt.xticks(rotation=fig_para.ticks_rotation, size=fig_para.tickssize)
        plt.yticks(size=fig_para.tickssize)
        if subtitle is None:
            plt.title(f"{i[0]}, {i[1]}", size=fig_para.titlesize)
            fig.savefig(
                f"{gl.get_value('OUTPUTPATH')}{i[0]},{i[1]}.jpg",
                dpi=fig_para.dpi,
                bbox_inches="tight",
                pad_inches=fig_para.pad_inches,
            )
        else:
            plt.title(f"{subtitle}_{i[0]},{i[1]}", size=fig_para.titlesize)
            fig.savefig(
                f"{gl.get_value('OUTPUTPATH')}{subtitle}_{i[0]},{i[1]}.jpg",
                dpi=fig_para.dpi,
                bbox_inches="tight",
                pad_inches=fig_para.pad_inches,
            )
        plt.clf()
    plt.clf()
    plt.close(fig)
    plt.close("all")


def draw_week_compare(
    datas: pd.DataFrame,
    xlabel: str = "time(week_hour)",
    ylabel: str = "kwh/1hr",
    dayhavedata: int = 24,
    time: str = "1hr",
    drawrange: str = "7day",
) -> None:
    """draw all meter's week value compare picture.

    Args:
        datas (pd.DataFrame): Draw data.
        xlabel (str, optional): X axis label. Defaults to "time(week_hour)".
        ylabel (str, optional): Y axis label. Defaults to "kwh/1hr".
        dayhavedata (int, optional): A few data a day. Defaults to 24.
        time (str, optional): Select to time resolution. Defaults to 1hr.
    """
    generate_outfile()
    sort_datas: DataFrame = datas.sort_values(
        by=["圖號座標", "組別", "cust_id", "meter_id", "read_time"], ignore_index=True
    )
    meter_id: list = list(sort_datas.groupby(["meter_id"]).groups)
    week: list = sort_datas.meter.complete_data_week(96 * 7 * len(meter_id))

    if time == "1hr":
        draw_vaule: DataFrame = sort_datas.meter.week_value
        matchstr: str = "1_\d\dhr"
        if drawrange == "1day":
            if dayhavedata == 24:
                dayhavedata = 3
    elif time == "15min":
        draw_vaule: DataFrame = sort_datas.meter.week_value_15min
        matchstr: str = "0_\d\d:\d\d"
        if dayhavedata == 24:
            if drawrange == "1day":
                dayhavedata = 3 * 4
            elif drawrange == "7day":
                dayhavedata == 24 * 4
    else:
        raise ValueError(f"No have this time: {time}")
    x: list = list(draw_vaule.columns)
    idx = pd.IndexSlice
    if drawrange == "7day":
        for meter in meter_id:
            with FigPara(
                title=f"{meter}",
                generate_newfile=f"week compare_7day_{time}",
                xlabel=xlabel,
                ylabel=ylabel,
                logger=logging.debug,
                show=False,
            ) as f:
                f.fig.gca().xaxis.set_major_locator(ticker.MultipleLocator(dayhavedata))
                drawlines: list = []
                p_label: list = []
                for week_for in week:
                    drawlines.append(x)
                    drawlines.append(draw_vaule.loc[idx[meter, week_for], :])
                    p_label.append(f"week {week_for}")
                temp = plt.plot(*drawlines, linewidth=f.linewidth)
                plt.legend(temp, p_label, prop={"size": f.legend_size}, loc="upper right")

    elif drawrange == "1day":
        onedaylist: list = []
        for ii, v in enumerate(x):
            match: Optional[re.Match] = re.match(matchstr, v)
            if bool(match):
                onedaylist.append(ii)
            onedaylong: int = len(onedaylist)
        for meter in meter_id:
            for weekday in range(int(len(x) / len(onedaylist))):
                with FigPara(  # 使用 with...as 來做流程控制 ; 一開始為了開畫圖視窗
                    title=f"{meter} weekday_{weekday+1}",
                    generate_newfile=f"week compare_weekday_{time}",
                    xlabel=xlabel,
                    ylabel=ylabel,
                    logger=logging.debug,
                ) as f:
                    f.fig.gca().xaxis.set_major_locator(
                        ticker.MultipleLocator(dayhavedata)
                    )  # 開視窗後的功能
                    drawlines: list = []
                    p_label: list = []
                    for week_for in week:
                        drawlines.append(
                            x[onedaylong * weekday : onedaylong + onedaylong * weekday]
                        )
                        drawlines.append(
                            draw_vaule.loc[
                                idx[meter, week_for],
                                draw_vaule.columns[onedaylong * weekday] : draw_vaule.columns[
                                    onedaylong + onedaylong * weekday - 1
                                ],
                            ]
                        )
                        p_label.append(f"week {week_for}")
                    temp: list = plt.plot(
                        *drawlines, linewidth=f.linewidth
                    )  # plot()前面加（*），一次畫出矩陣裡的所有曲線
                    plt.legend(temp, p_label, prop={"size": f.legend_size}, loc="upper right")
    else:
        raise ValueError(f"No have drawrange: {drawrange}")


def draw_oneday(
    meter_id: list,
    week: list,
    draw_vaule: pd.DataFrame,
    x: list,
    xlabel: str = "time(week_hour)",
    ylabel: str = "kwh/1hr",
    dayhavedata: int = 24,
    logger: Callable[[str], None] = logging.debug,
) -> None:
    i_l: list = []
    for ii, v in enumerate(x):
        # match: Optional[re.Match] = re.match("1_\d\dhr", v)
        match: Optional[re.Match] = re.match("0_\d\d:\d\d", v)
        if bool(match):
            i_l.append(ii)

    idx = pd.IndexSlice
    for i in meter_id:
        for jj in range(int(len(x) / len(i_l))):
            with FigPara(  # 使用 with...as 來做流程控制 ; 一開始為了開畫圖視窗
                title=f"{i} weekday_{jj+1}",
                generate_newfile="week compare_weekday_15min",
                xlabel=xlabel,
                ylabel=ylabel,
                logger=logger,
            ) as f:
                f.fig.gca().xaxis.set_major_locator(ticker.MultipleLocator(dayhavedata))  # 開視窗後的功能
                drawlines: list = []
                p_label: list = []
                for j in week:
                    drawlines.append(x[0 + len(i_l) * jj : len(i_l) + len(i_l) * jj])
                    drawlines.append(
                        draw_vaule.loc[
                            idx[i, j],
                            draw_vaule.columns[0 + len(i_l) * jj] : draw_vaule.columns[
                                len(i_l) + len(i_l) * jj - 1
                            ],
                        ]
                    )
                    p_label.append(f"week {j}")
                temp = plt.plot(*drawlines, linewidth=f.linewidth)  # plot()前面加（*），一次畫出矩陣裡的所有曲線
                plt.legend(temp, p_label, prop={"size": f.legend_size}, loc="upper right")


class NoTransformerCapacityError(Exception):
    """If no transfromer capacity, it will raise error."""


class NoAlignTimeError(Exception):
    """If can't align time, it will raise error."""


def load_uti_rate(
    coor: str,
    group: str,
    meter_data: pd.DataFrame,
    trans_data: pd.DataFrame,
    user_data: pd.DataFrame,
    logger: Optional[Callable[[str], None]] = logging.debug,
    Logger: Optional[logging.Logger] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    # if Logger:
    #     CopyLogger: logging.Logger = logging.getLogger(Logger.name)

    def log_mode(str: str, status: str) -> None:
        if logger:
            if Logger:
                if status == "error":
                    Logger.error(str)
                elif status == "warning":
                    Logger.warning(str)
                else:
                    Logger.debug(str)
            else:
                logger(str)

    transformer: DataFrame = trans_data.rename(
        columns={
            "第一具變壓器容量": "cap1",
            "第二具變壓器容量": "cap2",
            "第三具變壓器容量": "cap3",
            "用戶數": "user_amount",
            "主設備圖號座標": "coor",
            "組別": "group",
        }
    )
    transformer.set_index(["coor", "group"], inplace=True)
    transformer.sort_index(inplace=True)
    user: DataFrame = user_data.rename(
        columns={
            "表號": "cust_id",
            "相別": "type",
            "契約容量": "contract_cap",
            "圖號座標": "coor",
            "組別": "group",
        }
    )
    user.set_index(["coor", "group"], inplace=True)
    user.sort_index(inplace=True)
    # meter_data.meter.get_specific_cust_kwh("824343409")
    custom: DataFrame = meter_data.set_index(["cust_id"])
    custom.index = custom.index.astype(int)
    custom.sort_index(inplace=True)
    if logger:
        custom.out.log(name="custom", logger=logger)
        transformer.out.log(name="transformer", logger=logger)
        user.out.log(name="user", logger=logger)

    # idx = pd.IndexSlice
    # meter_type: list[str] = list(
    #     user.loc[(coor, group), :].groupby(["type"]).groups.keys()
    # )  # get this coordinate & group's meter_type.
    contract_cap: list[int] = list(
        user.loc[(coor, group), :].groupby(["contract_cap"]).groups.keys()
    )
    if not all([_ == 0 for _ in contract_cap]):  # Determine whether there is a contract user.
        havecontract: bool = True
    else:
        havecontract: bool = False
    time: list[pd.Timestamp] = list(meter_data.groupby(["read_time"]).groups.keys())
    time.sort()

    count: int = 0
    transformer_capacity: list[int] = list(transformer.loc[(coor, group), ("cap1", "cap2", "cap3")])
    for i in range(len(transformer_capacity)):
        if transformer_capacity[i] == 0:
            count += 1

    def _calculate_transformer_load() -> pd.DataFrame:
        transformer_load: DataFrame = pd.DataFrame(0, index=pd.Index(time), columns=["kwh"])
        cust: list[str] = list(user.loc[(coor, group), :].groupby(["cust_id"]).groups.keys())
        for c in cust:
            try:
                # value: DataFrame = custom.loc[c].sort_values(by=["read_time"])
                value: DataFrame = custom.loc[c].sort_values(by=["read_time"])
                try:
                    if all(value["read_time"] == time):
                        transformer_load += value["ratio_kwh"].values.reshape(
                            value["ratio_kwh"].values.size, 1
                        )
                    else:
                        raise NoAlignTimeError(
                            f"It can't to align time series: {coor}, {group}, {c}"
                        )
                except ValueError as e:
                    log_mode(
                        str=f"In {coor}, {group},{c} meter no have complete time data: {e}",
                        status="error",
                    )
                    transformer_load.reset_index(inplace=True)
                    transformer_load.rename(columns={"index": "read_time"}, inplace=True)
                    value.reset_index(inplace=True)
                    value.drop(columns=['cust_id', '圖號座標', '組別', 'meter_id'], inplace=True)
                    value.rename(columns={"ratio_kwh": "kwh"}, inplace=True)
                    transformer_load: DataFrame = (
                        pd.concat([transformer_load, value])
                        .groupby(['read_time'])
                        .sum()
                        .reset_index()
                    )
                    transformer_load.set_index('read_time', inplace=True)
                    transformer_load.index.name = None

            except KeyError as e:
                log_mode(str=f"In {coor}, {group}, no have {e} meter 15min data.", status="error")
        return transformer_load

    if count == 0:
        """Three phase transformer."""
        transformer_load: DataFrame = _calculate_transformer_load()
        if not all([_ == transformer_capacity[0] for _ in transformer_capacity]):
            log_mode(
                str=f"Three-phase transformer isn't have same capacity: {transformer_capacity}",
                status="warning",
            )
        transformer_uti_rate: DataFrame = (
            transformer_load * 4 * 100 / sum(transformer_capacity)
        )  # [(kwh/15min)*4/(t_cap*3)]*100
        transformer_uti_rate.rename(columns={"kwh": "rate"}, inplace=True)
        # transformer_uti_rate.plot(kind='line', y="rate")
        # plt.show()
        # transformer_uti_rate.out.log(name="uti rate", logger=logger.debug)
        if havecontract:
            """Have contract user's program."""
    elif count == 1:
        """U-V"""
        transformer_resident: DataFrame = pd.DataFrame(0, index=pd.Index(time), columns=["kwh"])
        transformer_industry: DataFrame = pd.DataFrame(0, index=pd.Index(time), columns=["kwh"])
        cust: list[str] = list(user.loc[(coor, group), :].groupby(["cust_id"]).groups.keys())
        empty_data: DataFrame = pd.DataFrame(0, index=pd.Index(time), columns=["ratio_kwh"])
        empty_data.index.name = "read_time"
        for c in cust:
            cust_type: object = (
                user.reset_index()
                .set_index(["coor", "group", "cust_id"])
                .loc[(coor, group, c), 'type']
            )
            try:
                value: DataFrame = custom.loc[c].sort_values(by=["read_time"])
                if not all(value["read_time"] == time):
                    raise NoAlignTimeError(f"It can't to align time series: {coor}, {group}, {c}")
            except ValueError as e:
                log_mode(f"In {coor}, {group},{c} meter no have complete time data: {e}", "error")
                # logging.error(f"In {coor}, {group},{c} meter no have complete time data: {e}")
                value = (
                    value.set_index("read_time").resample("15min").sum().sort_index().reset_index()
                )  # Resample time axis to every 15 minutes and fill in missing data with 0.
                if value.shape[0] != len(time):
                    value = (
                        pd.concat([value, empty_data.reset_index()])
                        .groupby("read_time")
                        .sum()
                        .sort_index()
                        .reset_index()
                    )
            except KeyError as e:
                log_mode(f"No have cust_id 15min data(uv phase): {e}", "error")
                # logging.error(f"No have cust_id 15min data(uv phase): {e}")
                value: DataFrame = pd.DataFrame(0, index=pd.Index(time), columns=["ratio_kwh"])

            if cust_type == "D" or cust_type == "B" or cust_type == "A":
                transformer_resident += value["ratio_kwh"].values.reshape(
                    value["ratio_kwh"].values.size, 1
                )
            else:
                if re.match('[0-9]{0,}.[0-9]{0,}kw', cust_type) is None:
                    transformer_industry += value["ratio_kwh"].values.reshape(
                        value["ratio_kwh"].values.size, 1
                    )
                else:
                    if value["ratio_kwh"].values.max() == 0:  # constract calculate.
                        temp: DataFrame = pd.DataFrame(
                            float(re.split('kw', cust_type)[0]) / 4,
                            index=pd.Index(time),
                            columns=["ratio_kwh"],
                        )
                        transformer_resident += temp.values.reshape(temp.values.size, 1)
                    else:  # ami data sum.
                        transformer_resident += value["ratio_kwh"].values.reshape(
                            value["ratio_kwh"].values.size, 1
                        )

        transformer_uti_rate_resident: DataFrame = (
            transformer_resident * 4 * 100 / max(transformer_capacity)
        )
        transformer_uti_rate_resident.rename(columns={"kwh": "rate"}, inplace=True)
        transformer_uti_rate_industry: DataFrame = (
            transformer_industry
            * 4
            * 100
            / min([transformer_capacity[_] for _, e in enumerate(transformer_capacity) if e != 0])
        )
        transformer_uti_rate_industry.rename(columns={"kwh": "rate"}, inplace=True)

        if havecontract:
            """Have contract user's program."""
    elif count == 2:
        """one phase transformer."""
        transformer_load: DataFrame = _calculate_transformer_load()
        one_phase_cap: list = [
            transformer_capacity[_] for _, e in enumerate(transformer_capacity) if e != 0
        ]
        transformer_uti_rate: DataFrame = (
            transformer_load * 4 * 100 / one_phase_cap[0]
        )  # [(kwh/15min)*4/(t_cap)]*100
        transformer_uti_rate.rename(columns={"kwh": "rate"}, inplace=True)
        # transformer_uti_rate.plot(kind='line', y="rate")
        # plt.show()
        # transformer_uti_rate.out.log(name="uti rate", logger=logging.debug)
        if havecontract:
            """Have contract user's program."""

    else:
        raise NoTransformerCapacityError(f"No have Transformer Capacity: {coor}, {group}")

    if count == 1:
        return transformer_uti_rate_resident, transformer_uti_rate_industry
    else:
        return transformer_uti_rate


# def draw_missing_point() -> None:
