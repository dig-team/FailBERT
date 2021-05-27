#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = Chadi Helwe 
__version__ = 1.0
__maintainer__ = Chadi Helwe
__email__ = chadi.helwe@telecom-paris.fr
__description__ = Module to create the dataset of natural parity task
"""

import csv
import itertools
import random


def create_dataset(
    path_natural_parity_dataset: str,
    min_range_length: int,
    max_range_length: int,
    min_nbr_switch_operation: int,
    max_nbr_switch_operation: int,
    random_sample: bool,
) -> None:
    """
    Create a parity dataset

    :param path_natural_parity_dataset: Path of the natural parity dataset
    :type path_natural_parity_dataset: str
    :param min_range_length: Minimum length of an instance
    :type min_range_length: int
    :param max_range_length: Maximum length of an instance
    :type max_range_length: int
    :param min_nbr_switch_operation: Minimum number of switch operations
    :type min_nbr_switch_operation: int
    :param max_nbr_switch_operation: Maximum number of switch operations
    :type max_nbr_switch_operation: int
    :param random_sample: If true the instances will be created randomly. Otherwise, the instances will be created using permutations
    :type random_sample: bool
    """

    list_pizzas = ["I ate a pizza"] * max_range_length
    list_switch = ["I operated the switch"] * max_range_length

    list_pizza_switch = list_pizzas + list_switch

    cnt = 0
    PERMUTATION_ELEMENTS = 50
    COMBINATION_ELEMENTS = 50
    cnt_true = 0

    with open(path_natural_parity_dataset, "w") as f:
        dataset_writer = csv.writer(f)
        dataset_writer.writerow(["modified_sentence", "label"])
        rows = []

        for it in range(min_range_length, max_range_length + 1):
            list_combination = []
            if random_sample:
                for _ in range(COMBINATION_ELEMENTS):
                    combination_instance = tuple(random.sample(list_pizza_switch, it))
                    list_combination.append(combination_instance)
            else:
                for i in itertools.combinations_with_replacement(list_pizza_switch, it):
                    list_combination.append(i)

            for i in set(list_combination):
                cnt_switch = i.count("I operated the switch")
                if (
                    cnt_switch >= min_nbr_switch_operation
                    and cnt_switch <= max_nbr_switch_operation
                ):
                    if cnt_switch % 2 != 0:
                        label = True
                    else:
                        label = False

                    list_permutation = []

                    if random_sample:
                        i = list(i)
                        for _ in range(PERMUTATION_ELEMENTS):
                            random.shuffle(i)
                            list_permutation.append(tuple(i))
                    else:
                        for e in itertools.permutations(i):
                            list_permutation.append(e)
                    set_permutation = set(list_permutation)

                    for e in set_permutation:
                        sentence = f"{' , and '.join(e[:])}"
                        sentence += " ."
                        rows.append(tuple([sentence, label]))

        for r in set(rows):
            dataset_writer.writerow(r)
            cnt += 1

            if r[1]:
                cnt_true += 1

    print(f"Total instances: {cnt}")
    print(f"Positive instances: {cnt_true}")
    print(f"Negative instances: {cnt - cnt_true}")
