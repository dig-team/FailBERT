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


cli.add_command(create_natural_dyck_2_dataset)

if __name__ == "__main__":
    cli()
