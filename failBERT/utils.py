from typing import Tuple

import numpy as np
import pandas as pd


def read_data(
    data_path: str, passages_column: str, labels_column: str
) -> Tuple[np.ndarray, np.ndarray]:
    """[summary]

    :param data_path: [description]
    :type data_path: str
    :param passages_column: [description]
    :type passages_column: str
    :param labels_column: [description]
    :type labels_column: str
    :return: [description]
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    data = pd.read_csv(data_path)
    passages = data[passages_column].values
    labels = data[labels_column].values
    return passages, labels
