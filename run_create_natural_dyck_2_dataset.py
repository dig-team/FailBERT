#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = Chadi Helwe
__version__ = 1.0
__maintainer__ = Chadi Helwe
__email__ = chadi.helwe@telecom-paris.fr
__description__ = CLI to create the natural dyck-2 dataset
"""

import click

from failBERT.create_natural_dyck_2_dataset import create_dataset
from failBERT.utils import create_equally_distributed_dataset, split_dataset


@click.group()
def cli():
    pass


@click.command()
@click.option("--path_dyck_2_dataset")
@click.option("--path_natural_dyck_2_dataset")
def create_natural_dyck_2_dataset(
    path_dyck_2_dataset: str, path_natural_dyck_2_dataset: str
) -> None:
    """
    Command to create a natural dyck-2 dataset

    :param path_dyck_2_dataset: Path of the dyck-2 dataset
    :type path_dyck_2_dataset: str
    :param path_natural_dyck_2_dataset: Path to save the natural dyck-2 dataset
    :type path_natural_dyck_2_dataset: str
    """

    create_dataset(path_dyck_2_dataset, path_natural_dyck_2_dataset)


@click.command()
@click.option("--path_dataset")
@click.option("--path_equally_distrbuted_dataset")
@click.option("--limit", default=True)
@click.option("--nbr_instances", default=10000)
def create_equally_distributed_natural_dyck_2_dataset(
    path_dataset: str,
    path_equally_distrbuted_dataset: str,
    limit: bool,
    nbr_instances: int,
) -> None:
    """
    Command to create an equally ditributed dataset from an unequally distributed dataset

    :param path_dataset: Path of the dataset
    :type path_dataset: str
    :param path_equally_distrbuted_dataset: Path to save the equally distributed dataset
    :type path_equally_distrbuted_dataset: str
    :param limit: If true the limitation is based on the next parameter. Otherwise, the limiation is based on the positive instances, defaults to True
    :type limit: bool, optional
    :param nbr_instances: Number of postive and negative instances, defaults to 10000
    :type nbr_instances: int, optional
    """

    create_equally_distributed_dataset(
        path_dataset, path_equally_distrbuted_dataset, limit, nbr_instances
    )


@click.command()
@click.option("--path_dataset")
@click.option("--path_train")
@click.option("--path_val")
@click.option("--path_test")
@click.option("--passages_column", default="modified_sentence")
@click.option("--labels_column", default="label")
@click.option("--upsample", default=False)
def split_natural_dyck_2_dataset(
    path_dataset: str,
    path_train: str,
    path_val: str,
    path_test: str,
    passages_column: str,
    labels_column: str,
    upsample: bool,
) -> None:
    """
    Command to split a dataset into training 60%, validation 20%, and testing 20% sets

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

    split_dataset(
        path_dataset,
        path_train,
        path_val,
        path_test,
        passages_column,
        labels_column,
        upsample,
    )


cli.add_command(create_natural_dyck_2_dataset)
cli.add_command(create_equally_distributed_natural_dyck_2_dataset)
cli.add_command(split_natural_dyck_2_dataset)

if __name__ == "__main__":
    cli()
