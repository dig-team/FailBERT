#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = Chadi Helwe
__version__ = 1.0
__maintainer__ = Chadi Helwe
__email__ = chadi.helwe@telecom-paris.fr
__description__ = CLI to run the natural parity experiment
"""

from typing import Optional

import click

from failBERT.eval import eval_model as eval_model_natural_parity
from failBERT.train import train_model as train_model_natural_parity


@click.group()
def cli():
    pass


@click.command()
@click.option(
    "--path_train",
    default="data/natural_parity/natural_parity_train.csv",
)
@click.option("--path_val", default=None)
@click.option("--passages_column", default="modified_sentence")
@click.option("--labels_column", default="label")
@click.option("--path_save_model", default="models/best_model_natural_parity.pkl")
@click.option("--epochs", default=10)
@click.option("--device", default="cpu")
def train_model(
    path_train: str,
    path_val: Optional[str],
    passages_column: str,
    labels_column: str,
    path_save_model: str,
    epochs: int,
    device: str,
):
    """
    Command to train a RoBERTa model on the natural parity task

    :param path_train: Path of the training dataset
    :type path_train: str
    :param path_val: Path of te validation dataset
    :type path_val: Optional[str]
    :param passage_column: Passage column name
    :type passage_column: str
    :param label_column: Label column name
    :type label_column: str
    :param path_save_model: Path to save the best model
    :type path_save_model: str
    :param epochs: Number of epochs
    :type epochs: int
    :param device: Device to run a model [GPU/CPU]
    :type device: str
    """
    train_model_natural_parity(
        path_train,
        path_val,
        passages_column,
        labels_column,
        path_save_model,
        epochs,
        device,
    )


@click.command()
@click.option(
    "--path_test",
    default="data/natural_parity/natural_parity_test.csv",
)
@click.option("--passages_column", default="modified_sentence")
@click.option("--labels_column", default="label")
@click.option("--path_model", default="models/best_model_natural_parity.pkl")
@click.option("--device", default="cpu")
def eval_model(
    path_test: str,
    passages_column: str,
    labels_column: str,
    path_model: str,
    device: str,
):
    """
    Command to evaluate a RoBERTa model on the natural parity task

    :param path_test: Path of the testing dataset
    :type path_test: str
    :param passage_column: Passage column name
    :type passage_column: str
    :param label_column: Label column name
    :type label_column: str
    :param path_model: Path of the saved model
    :type path_model: str
    :param device: Device to run a model [GPU/CPU]
    :type device: str
    """
    eval_model_natural_parity(
        path_test,
        passages_column,
        labels_column,
        path_model,
        device,
    )


cli.add_command(train_model)
cli.add_command(eval_model)

if __name__ == "__main__":
    cli()
