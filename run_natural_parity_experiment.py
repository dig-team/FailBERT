#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = Chadi Helwe
__version__ = 1.0
__maintainer__ = Chadi Helwe
__email__ = chadi.helwe@telecom-paris.fr
__description__ = Code to run the natural parity (light switch task) experiment in reported the paper
"""
import os
import warnings
from statistics import mean

from failBERT.create_natural_parity_dataset import create_dataset
from failBERT.eval import eval_model
from failBERT.utils import (create_equally_distributed_dataset,
                            download_pretrained_models)

warnings.filterwarnings("ignore")


def run_experiment(device: str) -> None:
    """[summary]

    :param device: Device to run a model [cpu/cuda]
    :type device: str
    """
    PATH_NATURAL_PARITY = "data/natural_parity/"
    MIN_LENGTH = 21
    MAX_LENGTH = 40
    MIN_SWITCH = 16
    MAX_SWITCH = 20
    NBR_INSTANCES = 2500
    MODEL_URL = "https://www.dropbox.com/s/c8ushxx3fow4yag/pizza_switch_best_model_1_15.pkl?dl=1"
    MODEL_NAME = "best_model_natural_parity.pkl"

    print("Creating datasets")
    print("######################################################")
    for i in range(1, 11):
        print(f"Test Dataset {i} created")
        path_temp_test_dataset = f"{PATH_NATURAL_PARITY}temp_test_{i}.csv"
        path_test_dataset = f"{PATH_NATURAL_PARITY}test_{i}.csv"
        create_dataset(
            path_temp_test_dataset, MIN_LENGTH, MAX_LENGTH, MIN_SWITCH, MAX_SWITCH, True
        )
        create_equally_distributed_dataset(
            path_temp_test_dataset, path_test_dataset, True, NBR_INSTANCES
        )
        print("######################################################")

    download_pretrained_models(MODEL_URL, MODEL_NAME)

    f1_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []

    print("Evalating the model")
    print("######################################################")
    natural_parity_dir = os.listdir(PATH_NATURAL_PARITY)
    for f in natural_parity_dir:
        if "test" in f:
            f1, accuracy, precision, recall = eval_model(
                f"{PATH_NATURAL_PARITY}{f}",
                "modified_sentence",
                "label",
                f"models/{MODEL_NAME}",
                device,
            )
            print("######################################################")
            f1_scores.append(f1)
            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)

    print(f"Average F1-Score: {mean(f1_scores)}")
    print(f"Average Accuracy Score: {mean(accuracy_scores)}")
    print(f"Average Precision Score: {mean(precision_scores)}")
    print(f"Average Recall Score: {mean(recall_scores)}")


if __name__ == "__main__":
    DEVICE = "cpu"  # or cuda
    run_experiment(DEVICE)
