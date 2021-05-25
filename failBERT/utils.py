#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = Chadi Helwe
__version__ = 1.0
__maintainer__ = Chadi Helwe
__email__ = chadi.helwe@telecom-paris.fr
__description__ = Utilities methods used by other modules
"""

from typing import Tuple

import numpy as np
import pandas as pd


def read_data(
    data_path: str, passages_column: str, labels_column: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a dataset by specifying the passages and the labels columns

    :param data_path: Path of a dataset
    :type data_path: str
    :param passages_column: Passages column name
    :type passages_column: str
    :param labels_column: Labels column name
    :type labels_column: str
    :return: Passages and labels
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    data = pd.read_csv(data_path)
    passages = data[passages_column].values
    labels = data[labels_column].values
    return passages, labels
