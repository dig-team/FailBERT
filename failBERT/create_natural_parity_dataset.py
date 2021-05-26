#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = Chadi Helwe 
__version__ = 1.0
__maintainer__ = Chadi Helwe
__email__ = chadi.helwe@telecom-paris.fr
__description__ = Module to create the dataset of natural parity task
"""

import itertools
import csv
import random


def create_dataset(
    path_natural_parity_dataset: str,
    min_range: int,
    max_range: int,
    random_sample: bool,
) -> None:
    """[summary]

    :param path_natural_parity_dataset: [description]
    :type path_natural_parity_dataset: str
    :param min_range: [description]
    :type min_range: int
    :param max_range: [description]
    :type max_range: int
    :param random_sample: [description]
    :type random_sample: bool
    """

    list_pizzas = ["I ate a pizza"] * max_range
    list_switch = ["I operated the switch"] * max_range

    list_pizza_switch = list_pizzas + list_switch

    cnt = 0
    PERMUTATION_ELEMENTS = 50
    COMBINATION_ELEMENTS = 50
    cnt_true = 0

    with open(path_natural_parity_dataset, "w") as f:
        dataset_writer = csv.writer(f)
        dataset_writer.writerow(["modified_sentence", "label"])
        rows = []

        for it in range(min_range, max_range + 1):
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
                if cnt_switch >= 1 and cnt_switch <= 15:
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

        print(len(rows))
        for r in set(rows):
            dataset_writer.writerow(r)
            cnt += 1

            if r[1]:
                cnt_true += 1

    print(cnt)
    print(cnt_true)
