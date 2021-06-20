#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = Chadi Helwe
__version__ = 1.0
__maintainer__ = Chadi Helwe
__email__ = chadi.helwe@telecom-paris.fr
__description__ = Code to run the natural dyck-2 (cake task) experiment in reported the paper
"""
import os
from statistics import mean

from failBERT.create_natural_dyck_2_dataset import create_dataset
from failBERT.eval import eval_model
from failBERT.utils import download_pretrained_models


def run_experiment(device: str) -> None:
    """
    Method to run the natural dyck-2 (cake task) experiment in reported the paper

    :param device: Device to run a model [cpu/cuda]
    :type device: str
    """
    PATH_DYCK_2 = "data/dyck_2/"
    PATH_NATURAL_DYCK_2 = "data/natural_dyck_2/"
    MODEL_URL = "https://www.dropbox.com/s/bxjmcrs7p737zfx/padded_best_model_swapped_natural_dyck.pkl?dl=1"
    MODEL_NAME = "best_model_natural_dyck_2.pkl"

    print("Creating datasets")
    print("######################################################")
    dyck_2_dir = os.listdir(PATH_DYCK_2)
    cnt = 1
    for f in dyck_2_dir:
        if "test" in f:
            print(f"Test dataset {cnt} created")
            create_dataset(
                f"{PATH_DYCK_2}{f}", f"{PATH_NATURAL_DYCK_2}natural_{f[:-4]}.csv"
            )
            print("######################################################")
            cnt += 1

    download_pretrained_models(MODEL_URL, MODEL_NAME)

    f1_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []

    print("Evalating the model")
    print("######################################################")
    natural_dyck_2_dir = os.listdir(PATH_NATURAL_DYCK_2)
    for f in natural_dyck_2_dir:
        if "test" in f:
            f1, accuracy, precision, recall = eval_model(
                f"{PATH_NATURAL_DYCK_2}{f}",
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
