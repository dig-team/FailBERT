from statistics import mean

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from failBERT.dataloader import CustomDataset


def eval_model(
    path_test: str,
    passages_column: str,
    labels_column: str,
    path_model: str,
    device: str,
):
    """
    Evaluate a saved RoBERTa model on a testing dataset

    :param path_test: Path of a testing dataset
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

    best_model = torch.load(path_model)

    best_model.to(device)

    test_dataset = CustomDataset(path_test, passages_column, labels_column)
    test_dataloader = DataLoader(test_dataset, batch_size=32)

    best_model.eval()
    test_f1_scores = []
    test_accuracy_scores = []
    all_y_pred = []
    all_y_true = []

    with torch.no_grad():
        for _, x, x_attention, y in tqdm(test_dataloader):
            input_ids = x["input_ids"].squeeze()[:, :-1]
            outputs = best_model(
                input_ids=input_ids.to(device),
                attention_mask=x_attention.to(device),
                labels=y.to(device),
            )

            y_pred = torch.argmax(outputs.logits, dim=1)
            y_pred = y_pred.detach().cpu().numpy()
            y_true = y.detach().cpu().numpy()

            all_y_pred.append(y_pred)
            all_y_true.append(y_true)

            test_f1_scores.append(f1_score(y_true, y_pred, average="micro"))
            test_accuracy_scores.append(accuracy_score(y_true, y_pred))

        avg_f1_scores = mean(test_f1_scores)
        avg_accuracy_scores = mean(test_accuracy_scores)

        print(f"Test F1 Score {avg_f1_scores}")
        print(f"Test Accuracy Score {avg_accuracy_scores}")
