#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = Chadi Helwe
__version__ = 1.0
__maintainer__ = Chadi Helwe
__email__ = chadi.helwe@telecom-paris.fr
__description__ = Utilities methods used by other modules
"""

import csv
import os
from collections import Counter
from typing import Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def read_dataset(
    path_dataset: str, passages_column: str, labels_column: str
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

    data = pd.read_csv(path_dataset)
    passages = data[passages_column].values
    labels = data[labels_column].values
    return passages, labels


def split_dataset(
    path_dataset: str,
    path_train: str,
    path_val: str,
    path_test: str,
    passages_column: str,
    labels_column: str,
    upsample: bool = False,
) -> None:
    """
    Split a dataset into training 60%, validation 20%, and testing 20% sets

    :param path_dataset: Path of the dataset
    :type path_dataset: str
    :param path_train: Path to save the training set
    :type path_train: str
    :param path_val: Path to save the validation set
    :type path_val: str
    :param path_test: Path to save the testing set
    :type path_test: str
    :param passages_column: Passages column name
    :type passages_column: str
    :param labels_column: Labels column name
    :type labels_column: str
    :param upsample: Upsample the training set for data augmentation, defaults to False
    :type upsample: bool, optional
    """

    data = pd.read_csv(path_dataset)
    data_train, data_temp, label_train, label_temp = train_test_split(
        data[passages_column],
        data[labels_column],
        test_size=0.4,
        stratify=data[labels_column],
    )

    data_val, data_test, label_val, label_test = train_test_split(
        data_temp,
        label_temp,
        test_size=0.5,
        stratify=label_temp,
    )

    with open(path_train, "w") as f:
        cnt = Counter(label_train)
        upsample_value = int(cnt[False] / cnt[True])
        csv_train = csv.writer(f)
        csv_train.writerow([passages_column, labels_column])
        if upsample:
            for i, l in zip(data_train, label_train):
                if l:
                    for _ in range(0, upsample_value):
                        csv_train.writerow([i, l])
                else:
                    csv_train.writerow([i, l])
        else:
            for i, l in zip(data_train, label_train):
                csv_train.writerow([i, l])

    with open(path_val, "w") as f:
        csv_val = csv.writer(f)
        csv_val.writerow([passages_column, labels_column])

        for i, l in zip(data_val, label_val):
            csv_val.writerow([i, l])

    with open(path_test, "w") as f:
        csv_test = csv.writer(f)
        csv_test.writerow([passages_column, labels_column])

        for i, l in zip(data_test, label_test):
            csv_test.writerow([i, l])


def count_pattern_dataset(
    path_dataset: str,
    passages_column: str,
    labels_column: str,
    pattern: str,
    label: bool,
) -> None:
    """
    Count the number of instances with a specific pattern and a label

    :param path_dataset: Path of the dataset
    :type path_dataset: str
    :param passages_column: Passages column name
    :type passages_column: str
    :param labels_column: Labels column name
    :type labels_column: str
    :param pattern: Pattern to check in the passages
    :type pattern: str
    :param label: Label associated with the searched pattern
    :type label: bool
    """

    data = pd.read_csv(path_dataset)
    cnt_pattern_label = 0
    for x, y in zip(data[passages_column], data[labels_column]):
        if pattern in x and y == label:
            cnt_pattern_label += 1
    print(
        f"Counts of passages with pattern {pattern} and label {label}: {cnt_pattern_label}"
    )


def create_equally_distributed_dataset(
    path_dataset: str,
    path_equally_distrbuted_dataset: str,
    limit: bool = True,
    nbr_instances: int = 10000,
) -> None:
    """
    Create an equally ditributed dataset from an unequally distributed dataset

    :param path_dataset: Path of the dataset
    :type path_dataset: str
    :param path_equally_distrbuted_dataset: Path to save the equally distributed dataset
    :type path_equally_distrbuted_dataset: str
    :param limit: If true the limitation is based on the next parameter. Otherwise, the limiation is based on the positive instances, defaults to True
    :type limit: bool, optional
    :param nbr_instances: Number of postive and negative instances, defaults to 10000
    :type nbr_instances: int, optional
    """

    data = pd.read_csv(path_dataset)
    if limit:
        x_true = data[data["label"] == True].sample(nbr_instances)
        x_false = data[data["label"] == False].sample(nbr_instances)
    else:
        x_true = data[data["label"] == True]
        x_false = data[data["label"] == False].sample(len(x_true))
        nbr_instances = len(x_true)

    x_true.reset_index(drop=True, inplace=True)
    x_false.reset_index(drop=True, inplace=True)
    new_data = pd.concat([x_true, x_false])

    new_data.to_csv(f"{path_equally_distrbuted_dataset}")
    print(
        f"Dataset was equally distributed with {nbr_instances} instances for each label"
    )


def download_pretrained_models(url: str, file_name: str) -> None:
    """
    Download a pretrained model for the tasks

    :param url: DropBox url of the pretrained model
    :type url: str
    :param file_name: Name of the pretrained model
    :type file_name: str
    """
    PATH_MODELS = "models"
    if not os.path.isdir(PATH_MODELS):
        os.mkdir(PATH_MODELS)

    if not os.path.exists(f"{PATH_MODELS}/{file_name}"):
        with requests.get(url, stream=True) as req:
            total_size_in_bytes = int(req.headers.get("content-length", 0))
            chunk_size = 1024
            progress_bar = tqdm(
                total=total_size_in_bytes,
                unit_scale=True,
                desc="Downloading Pretrained Model",
            )
            with open(f"{PATH_MODELS}/{file_name}", "wb") as f:
                for chunk in req.iter_content(chunk_size=chunk_size):
                    progress_bar.update(len(chunk))
                    f.write(chunk)
            progress_bar.close()
