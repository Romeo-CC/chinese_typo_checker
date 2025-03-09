import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW

from models.networks import HZTypoChecker
from utils.datahelper import StreamingTextDataset
from tqdm import tqdm
import numpy as np
from datetime import datetime


def timestamp():
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def save_model(model: HZTypoChecker, save_path: str):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model.corrector.save_pretrained(f"{save_path}/mlm")
    torch.save(model.detector.state_dict(), f"{save_path}/detector.pth")
            

def loss_function( 
        mlm_gt: torch.Tensor, cls_gt: torch.Tensor, 
        mlm_pred: torch.Tensor, cls_pred: torch.Tensor,
        config
    ) -> torch.Tensor:
    # ignore [PAD] tokens
    mlm_loss_fn = nn.CrossEntropyLoss()
    cls_loss_fn = nn.BCEWithLogitsLoss()

    mlm_loss = mlm_loss_fn(mlm_pred.view(-1, config.vocab_size), mlm_gt.view(-1))
    cls_loss = cls_loss_fn(cls_pred.view(-1), cls_gt.view(-1))

    return mlm_loss + cls_loss


def train_step(model, data_batch, optimizer, device):
    input_ids = data_batch["input_ids"].to(device)
    mlm_labels = data_batch["mlm_labels"].to(device)
    cls_labels = data_batch["cls_labels"].to(device)
    attention_mask = data_batch["attention_mask"].to(device)
    optimizer.zero_grad()
    mlm_pred, cls_pred = model(input_ids, attention_mask)
    loss = loss_function(mlm_labels, cls_labels, mlm_pred, cls_pred, model.config)
    loss.backward()
    optimizer.step()

    return loss.clone().detach().cpu().item()



def train(
        epochs, data_path, save_path,
        model_name, device, learning_rate,
        log_dir, batch_size=4
    ):
    
    model = HZTypoChecker(model_name=model_name)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    dataset = StreamingTextDataset(data_dir=data_path, tokenizer_name=model_name)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    writer = SummaryWriter(log_dir)
    gstep = 0
    for epoch in range(epochs):
        training_loss = []
        for batch in tqdm(dataloader, 
            desc=f"Training HZTypoChecker in epoch {epoch + 1}: "):

            loss_val = train_step(
                model=model, data_batch=batch, 
                optimizer=optimizer, device=device
            )
            training_loss.append(loss_val)
            if len(training_loss) == 1000:
                gstep += 1000
                loss_val = np.mean(training_loss)
                writer.add_scalar("HZTypoChecker Training Loss", loss_val, global_step=gstep)

                training_loss = []
        
        tail = len(training_loss)
        if tail != 0:
            gstep += tail
            loss_val = np.mean(training_loss)
            writer.add_scalar("HZTypoChecker Training Loss", loss_val, global_step=gstep)

            training_loss = []

        save_dir = f"{save_path}/{epoch}"
        save_model(model, save_dir)


data_path = "data/txt/trial"
log_dir = f"logs/{timestamp()}"
save_path = "weights"
model_name = "data/bert"

epochs = 5
learning_rate = 2e-5
batch_size = 32

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
if torch.mps.is_available():
    device = torch.device("mps")


train(
    epochs=epochs,
    data_path=data_path,
    save_path=save_path,
    model_name=model_name,
    device=device,
    learning_rate=learning_rate,
    log_dir=log_dir,
    batch_size=batch_size
)