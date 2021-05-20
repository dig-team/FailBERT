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
        """[summary]

        :param data_path: [description]
        :type data_path: str
        :param passages_column: [description]
        :type passages_column: str
        :param labels_column: [description]
        :type labels_column: str
        """
        self.passages, self.labels = read_data(
            data_path, passages_column, labels_column
        )
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def preprocess(
        self, passage: str, label: str
    ) -> Tuple[str, RobertaTokenizer, torch.Tensor, torch.Tensor]:
        """[summary]

        :param passage: [description]
        :type passage: str
        :param label: [description]
        :type label: str
        :return: [description]
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

    def __len__(self):
        """[summary]

        :return: [description]
        :rtype: [type]
        """
        return len(self.labels)

    def __getitem__(self, index: int):
        """[summary]

        :param index: [description]
        :type index: int
        :return: [description]
        :rtype: [type]
        """
        return self.preprocess(self.passages[index], self.labels[index])
