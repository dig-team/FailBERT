#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = Chadi Helwe
__version__ = 1.0
__maintainer__ = Chadi Helwe
__email__ = chadi.helwe@telecom-paris.fr
__description__ = CLI to create the natural parity dataset
"""

import click

from failBERT.create_natural_parity_dataset import create_dataset


@click.group()
def cli():
    pass


@click.command()
@click.option("--path_natural_parity_dataset")
@click.option("--min_range_length")
@click.option("--max_range_length")
@click.option("--min_nbr_switch_operation")
@click.option("--max_nbr_switch_operation")
@click.option("--random_sample", default=True)
def create_natural_parity_dataset(
    path_natural_parity_dataset: str,
    min_range_length: int,
    max_range_length: int,
    min_nbr_switch_operation: int,
    max_nbr_switch_operation: int,
    random_sample: bool,
) -> None:
    """
    Command to create a parity dataset

    :param path_natural_parity_dataset: Path to save the natural parity dataset
    :type path_natural_parity_dataset: str
    :param min_range_length: Minimum length of an instance
    :type min_range_length: int
    :param max_range_length: Maximum length of an instance
    :type max_range_length: int
    :param min_nbr_switch_operation: Minimum number of switch operations
    :type min_nbr_switch_operation: int
    :param max_nbr_switch_operation: Maximum number of switch operations
    :type max_nbr_switch_operation: int
    :param random_sample: If true the instances are created randomly. Otherwise, the instances are created using permutations
    :type random_sample: bool
    """

    create_dataset(
        path_natural_parity_dataset,
        min_range_length,
        max_range_length,
        min_nbr_switch_operation,
        max_nbr_switch_operation,
        random_sample,
    )


cli.add_command(create_natural_parity_dataset)

if __name__ == "__main__":
    cli()
