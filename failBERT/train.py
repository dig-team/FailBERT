from torch.utils.data import DataLoader
from failBERT.dataloader import CustomDataset
import torch
import torch.optim as optim
from transformers import RobertaForSequenceClassification
from tqdm import tqdm

from sklearn.metrics import f1_score
from statistics import mean 

def train(epochs:int, device:str):

    model = RobertaForSequenceClassification.from_pretrained('roberta-base')
    
    model.to(device)

    train_dataset = CustomDataset("data/natural_dyck_2_datasets/swapped_natural_dyck_train.csv", "modified_sentence", "label")
    val_dataset = CustomDataset("data/natural_dyck_2_datasets/swapped_natural_dyck_val.csv", "modified_sentence", "label")
    
    train_dataloader = DataLoader(train_dataset, batch_size=8,shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32,shuffle=False)
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-6)

    best_val_loss = 1000
    best_val_f1_score = 0

    for epoch in range(0, epochs):
        print("Epoch {}".format(epoch + 1))
        train_loss = 0
        val_loss = 0
        train_f1_scores = []
        val_f1_scores = []
        
        model.train()
        for _, x,x_attention, y in tqdm(train_dataloader):
            optimizer.zero_grad()
            input_ids = x["input_ids"].squeeze()[:,:-1]
            outputs = model(input_ids=input_ids.to(device), attention_mask=x_attention.to(device), labels=y.to(device))
            
            y_pred = torch.argmax(outputs.logits, dim=1)
            y_pred = y_pred.detach().cpu().numpy() 
            y_true = y.detach().cpu().numpy()
            
            train_f1_scores.append(f1_score(y_true, y_pred, average="micro"))
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_dataloader)
        print("Training loss {}".format(train_loss))
        print("Training F1-Score {}".format(mean(train_f1_scores)))
        
        model.eval()
        with torch.no_grad():
            for _, x,x_attention, y in tqdm(val_dataloader):
                input_ids = x["input_ids"].squeeze()[:,:-1]
                outputs = model(input_ids=input_ids.to(device), attention_mask=x_attention.to(device), labels=y.to(device))

                y_pred = torch.argmax(outputs.logits, dim=1)
                y_pred = y_pred.detach().cpu().numpy() 
                y_true = y.detach().cpu().numpy()
                
                val_f1_scores.append(f1_score(y_true, y_pred, average="micro"))
                
                loss = outputs.loss
                val_loss += loss.item()

            val_loss /= len(val_dataloader)
            
            if best_val_loss > val_loss:
                print("###########################")
                print("Best Model")
                best_val_loss = val_loss
                best_val_f1_score = mean(val_f1_scores)
                torch.save(model, "models/padded_best_model_swapped_natural_dyck.pkl")
                print("###########################")
                
            print(f"Val loss {val_loss}")
            print(f"Val F1-Score {mean(val_f1_scores)}")
    print(f"Best Val F1-Score {best_val_f1_score}")