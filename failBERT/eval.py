from statistics import mean

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from failBERT.dataloader import CustomDataset


def eval(
    path_test: str, passage_column: str, label_column: str, path_model: str, device: str
):
    """[summary]

    :param path_test: [description]
    :type path_test: str
    :param passage_column: [description]
    :type passage_column: str
    :param label_column: [description]
    :type label_column: str
    :param path_model: [description]
    :type path_model: str
    :param device: [description]
    :type device: str
    """
    MIN_NOT = 16
    MAX_NOT = 17
    # best_model = torch.load("models/padded_best_model_swapped_natural_dyck.pkl")

    best_model = torch.load(f"{path_model}")

    all_f1_scores = []
    best_model.to(device)
    for d in range(MIN_NOT, MAX_NOT):

        test_dataset = CustomDataset(
            "data/natural_dyck_2_datasets/swapped_natural_dyck_test_{}_20.csv".format(
                d
            ),
            "modified_sentence",
            "label",
        )
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

            all_f1_scores.append(avg_f1_scores)

            print("{} -> Test F1 Score {}".format(d, avg_f1_scores))
            print("{} -> Test Accuracy Score {}".format(d, avg_accuracy_scores))
