#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = Chadi Helwe 
__version__ = 1.0
__maintainer__ = Chadi Helwe
__email__ = chadi.helwe@telecom-paris.fr
__description__ = Module to create the dataset of natural dyck-2 task
"""

import csv
import random
from typing import List, Optional, Tuple


def convert_dyck_to_natural_instance(dyck_2: str) -> Tuple[List[str], List[str]]:
    """[summary]

    :param dyck_2: [description]
    :type dyck_2: str
    :return: [description]
    :rtype: Tuple[List[str], List[str]]
    """

    symbols_to_instance = {
        "[": "I added a peanut layer to my cake",
        "(": "I added a chocolate layer to my cake",
        "]": "I ate the peanut layer",
        ")": "I ate the chocolate layer",
    }

    list_str_dyck = []
    list_str_dyck_symbols = []
    for s in dyck_2:
        list_str_dyck.append(f"{symbols_to_instance[s]}")
        list_str_dyck_symbols.append(f"{s}: {symbols_to_instance[s]}")

    return list_str_dyck, list_str_dyck_symbols


def convert_to_swapped_false_instance(
    list_str_dyck: List[str], list_str_dyck_symbols: List[str]
) -> Tuple[List[str], List[str], Optional[int], bool]:
    """[summary]

    :param list_str_dyck: [description]
    :type list_str_dyck: List[str]
    :param list_str_dyck_symbols: [description]
    :type list_str_dyck_symbols: List[str]
    :return: [description]
    :rtype: Tuple[List[str], List[str], Optional[int], bool]
    """

    peanut_indexes = [
        i for i, s in enumerate(list_str_dyck) if s == "I ate the peanut layer"
    ]
    chocolate_indexes = [
        i for i, s in enumerate(list_str_dyck) if s == "I ate the chocolate layer"
    ]

    label = True
    index = None

    if len(peanut_indexes) > 0 and len(chocolate_indexes) > 0:
        index_1 = random.choice(peanut_indexes)
        index_2 = random.choice(chocolate_indexes)

        list_str_dyck[index_1], list_str_dyck[index_2] = (
            list_str_dyck[index_2],
            list_str_dyck[index_1],
        )
        list_str_dyck_symbols[index_1], list_str_dyck_symbols[index_2] = (
            list_str_dyck_symbols[index_2],
            list_str_dyck_symbols[index_1],
        )

        label = False
        index = (index_1, index_2)

    return list_str_dyck, list_str_dyck_symbols, index, label


def to_str(list_str_dyck: List[str]) -> str:
    """[summary]

    :param list_str_dyck: [description]
    :type list_str_dyck: List[str]
    :return: [description]
    :rtype: str
    """

    str_dyck = " , ".join(list_str_dyck[:-1])
    str_dyck += f" and {list_str_dyck[-1]} ."
    return str_dyck


def create_dataset(path_dyck_2_dataset: str, path_natural_dyck_2_dataset: str) -> None:
    """[summary]

    :param path_dyck_2_dataset: [description]
    :type path_dyck_2_dataset: str
    :param path_natural_dyck_2_dataset: [description]
    :type path_natural_dyck_2_dataset: str
    """

    with open(path_natural_dyck_2_dataset, "w") as dataset:
        dataset_writer = csv.writer(dataset)
        dataset_writer.writerow(
            [
                "modified_sentence",
                "modified_sentence_with_symbols",
                "dyck_format",
                "index",
                "label",
            ]
        )
        with open(path_dyck_2_dataset, "r") as f:

            for l in f.readlines():
                list_str_dyck, list_str_dyck_symbols = convert_dyck_to_natural_instance(
                    l.strip()
                )
                dataset_writer.writerow(
                    [
                        to_str(list_str_dyck),
                        to_str(list_str_dyck_symbols),
                        l.strip(),
                        str(None),
                        True,
                    ]
                )
                (
                    false_list_str_dyck,
                    false_list_str_dyck_symbols,
                    index,
                    label,
                ) = convert_to_swapped_false_instance(
                    list_str_dyck, list_str_dyck_symbols
                )
                if not label:
                    dataset_writer.writerow(
                        [
                            to_str(false_list_str_dyck),
                            to_str(false_list_str_dyck_symbols),
                            l.strip(),
                            index,
                            label,
                        ]
                    )
