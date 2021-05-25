#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = Chadi Helwe
__version__ = 1.0
__maintainer__ = Chadi Helwe
__email__ = chadi.helwe@telecom-paris.fr
__description__ = Module to load and preprocess data
"""

from typing import Tuple

import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

from failBERT.utils import read_data


class CustomDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        passages_column: str,
        labels_column: str,
    ):
        """
        :param data_path: Path of a dataset
        :type data_path: str
        :param passages_column: Passages column name
        :type passages_column: str
        :param labels_column: Labels column name
        :type labels_column: str
        """
        self.passages, self.labels = read_data(
            data_path, passages_column, labels_column
        )
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def preprocess(
        self, passage: str, label: str
    ) -> Tuple[str, RobertaTokenizer, torch.Tensor, torch.Tensor]:
        """
        Preprocess a passage

        :param passage: Passage
        :type passage: str
        :param label: Label
        :type label: str
        :return: Passage with its preprocessed passage, its attention mask, and its label
        :rtype: Tuple[str, RobertaTokenizer, torch.Tensor, torch.Tensor]
        """
        new_passage = passage.strip() + " </s>"
        len_new_passage = len(self.tokenizer.encode(new_passage)) - 1
        if len_new_passage < 512:
            new_passage += "<pad>" * (500 - len_new_passage)
        else:
            print(" > 512")
        new_label = None
        if label == True:
            new_label = 1
        else:
            new_label = 0

        attention_mask = [1] * len_new_passage + [0] * (500 - len_new_passage)
        return (
            passage,
            self.tokenizer(new_passage, return_tensors="pt"),
            torch.tensor(attention_mask),
            torch.tensor(new_label).unsqueeze(0),
        )

    def __len__(self) -> int:
        """
        :return: Length of the dataset
        :rtype: int
        """
        return len(self.labels)

    def __getitem__(
        self, index: int
    ) -> Tuple[str, RobertaTokenizer, torch.Tensor, torch.Tensor]:
        """
        :param index: Index of an instance in the dataset
        :type index: int
        :return:  Passage with its preprocessed passage, its attention mask, and its label
        :rtype: Tuple[str, RobertaTokenizer, torch.Tensor, torch.Tensor]
        """
        return self.preprocess(self.passages[index], self.labels[index])
