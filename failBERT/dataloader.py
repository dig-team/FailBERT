from typing import Tuple
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

from .utils import read_data


class CustomDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        passages_column: str,
        labels_column: str,
        tokenizer: RobertaTokenizer,
    ):
        self.passages, self.labels = read_data(
            data_path, passages_column, labels_column
        )
        self.tokenizer = tokenizer.from_pretrained("roberta-base")

    def preprocess(self, passage: str, label: str) -> Tuple[str, RobertaTokenizer, torch.Tensor, torch.Tensor]:
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
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.preprocess(self.passages[index], self.labels[index])
