#!/usr/bin/env python
# coding=utf-8
"""
File Description: Cross file global variable.
Author          : CHEN, JIA-LONG
Create Date     : 2022-12-30 20:46
FilePath        : \\N_20221213_ami_data_analyze\\globalvar.py
Copyright © 2023 CHEN JIA-LONG.
"""

import logging
from typing import Any, Optional


class ValueNotInGlobal(Exception):
    """When variable is not in globalvar.py, it's will be raise ValueNotInGloball.

    Args:
        name -- variable which caused th error
        message -- explantion of the error
    """

    def __init__(self, name: str) -> None:
        self.message: str = f"{name} is not define in global variable"
        super().__init__(self.message)


def _init() -> None:
    """
    Initialize globalvar.py code.
    When main program start, you must call globalvar._init()
    """
    global _global_dict
    _global_dict = {}


def set_value(name: str, value: Any, readonly: bool = False) -> None:
    """Set global variable.

    Args:
        name (str): variable name.
        value (Any): variable's value.
        readonly (bool, optional): When set true, it will be read only. Defaults to False.
    """
    if name in _global_dict.keys():
        if _global_dict[name][1] is True:
            logging.warning(f"{name} is only read, value is {_global_dict[name][0]}")
        else:
            _global_dict[name] = value, readonly
    else:
        _global_dict[name] = value, readonly


def get_value(name: str, defvalue: Any = None) -> Any:
    """Get global variable.

    Args:
        name (str): variable name.
        defvalue (any, optional):If global variable isn't exist, then retrun default value.
                                 Defaults to None.

    Raises:
        ValueNotInGlobal: If global variable isn't exist and default value is none,
                          then raise Error.

    Returns:
        Any: global variable.
    """
    try:
        return _global_dict[name][0]
    except KeyError:
        if defvalue is None:
            logging.error(f"{name} is not in global.")
            raise ValueNotInGlobal(name)
        else:
            logging.error(f"{name} is not in global, return default value.")
            return defvalue


def check_value(name: str) -> bool:
    """Check variable exist?

    Args:
        name (str): Need check variable name.

    Returns:
        bool: Ture means have variabel.
    """
    if name in _global_dict.keys():
        return True
    else:
        return False
