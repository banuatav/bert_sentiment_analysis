import transformers 
from transformers import (
    BertModel,
    BertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
import torch

import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, shuffle
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SenAnDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )

        return {
            "text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def load_data(df, tokenizer, max_len, batch_size, y_label, x_label):
    ds = SenAnDataset(
        texts=df[x_label].to_numpy(),
        labels=df[y_label].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len,
    )

    return DataLoader(ds, batch_size=batch_size, num_workers=4)


class SentimentClassifier(nn.Module):
    def __init__(self, n_classes, PRE_TRAINED_MODEL_NAME):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(pooled_output)
        return self.out(output)


def train_one_epoch(
    model, data_loader, scheduler, n_examples, loss_fn, optimizer, device=DEVICE,
):
    model = model.train()

    loss_list = []
    cor_pred = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        cor_pred += torch.sum(preds == labels)
        loss_list.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return cor_pred.double() / n_examples, np.mean(loss_list)


def evaluate_model(model, data_loader, n_examples, loss_fn, device=DEVICE):
    model = model.eval()

    loss_list = []
    cor_pred = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, labels)

            cor_pred += torch.sum(preds == labels)
            loss_list.append(loss.item())

    return cor_pred.double() / n_examples, np.mean(loss_list)


def prep_data(
    df,
    device=DEVICE,
    config={
        "PRE_TRAINED_MODEL_NAME": "bert-base-cased",
        "MAX_LEN": 25,
        "BATCH_SIZE": 16,
        "TRAIN_SIZE": 0.9,
        "Y_LABEL": "sentiment",
        "X_LABEL": "spans_combined",
        "RANDOM_SEED": 42,
        "RESAMPLE_IMBALANCED_CLASSES": True,
    },
):

    PRE_TRAINED_MODEL_NAME = config["PRE_TRAINED_MODEL_NAME"]
    MAX_LEN = config["MAX_LEN"]
    BATCH_SIZE = config["BATCH_SIZE"]
    TRAIN_SIZE = config["TRAIN_SIZE"]
    Y_LABEL = config["Y_LABEL"]
    X_LABEL = config["X_LABEL"]
    RANDOM_SEED = config["RANDOM_SEED"]
    RESAMPLE_IMBALANCED_CLASSES = config["RESAMPLE_IMBALANCED_CLASSES"]

    # Define train-val-test split
    print("Defining train-val-test split...")
    print()

    df_train, df_val = train_test_split(
        df, test_size=1 - TRAIN_SIZE, random_state=RANDOM_SEED
    )

    # Upsample negative class
    if RESAMPLE_IMBALANCED_CLASSES:
        df_neg = df_train[df_train[Y_LABEL] == 0]
        df_pos = df_train[df_train[Y_LABEL] == 1]
        df_neg_upsampled = resample(
            df_neg, replace=True, n_samples=len(df_pos), random_state=RANDOM_SEED
        )
        df_pos = resample(
            df_pos, replace=False, n_samples=len(df_pos), random_state=RANDOM_SEED
        )
        df_train = df_pos.append(df_neg_upsampled)
        df_train = shuffle(df_train).reset_index(drop=True)

    # Define tokenizer
    print("Defining tokenizer...")
    print()
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    # Create data loaders
    print("Creating data loaders...")
    print()
    load_train_data = load_data(
        df_train, tokenizer, MAX_LEN, BATCH_SIZE, Y_LABEL, X_LABEL
    )
    load_val_data = load_data(
        df_val, tokenizer, MAX_LEN, BATCH_SIZE, Y_LABEL, X_LABEL)

    data_sets = (df_train, df_val)
    loaders = (load_train_data, load_val_data)
    return data_sets, loaders, tokenizer


def train_bert(
    data_sets,
    loaders,
    loss_fn=nn.CrossEntropyLoss(),
    device=DEVICE,
    config={
        "PRE_TRAINED_MODEL_NAME": "bert-base-cased",
        "EPOCHS": 10,
        "LEARNING_RATE": 2e-5,
    },
):

    PRE_TRAINED_MODEL_NAME = config["PRE_TRAINED_MODEL_NAME"]
    EPOCHS = config["EPOCHS"]
    LEARNING_RATE = config["LEARNING_RATE"]

    print("-- Device selected for computations = ", DEVICE)
    print("-- Running training under following parameters: \n")
    print("-- LEARNING_RATE: {}".format(LEARNING_RATE))
    print("-- EPOCHS: {}".format(EPOCHS))
    print()

    # Unpack data
    print("-- Unpacking data...")
    print()
    load_train_data, load_val_data = loaders
    df_train, df_val = data_sets

    # Create classifier object
    print("-- Defining classifier...")
    print()
    model = SentimentClassifier(
        len(set(df_train["sentiment"])), PRE_TRAINED_MODEL_NAME)
    model = model.to(device)

    # Determine optimization parameters
    total_steps = len(load_train_data) * EPOCHS

    print("-- Defining optimization parameters...")
    print()

    loss_fn = loss_fn.to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # Start training
    print("-" * 10)
    print("START TRAINING")
    print("-" * 10)
    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(EPOCHS):
        print(f"-- Epoch {epoch + 1}/{EPOCHS}")
        print("-" * 10)

        train_acc, train_loss = train_one_epoch(
            model, load_train_data, scheduler, len(
                df_train), loss_fn, optimizer
        )

        print(f"-- Train loss {train_loss} accuracy {train_acc}")

        val_acc, val_loss = evaluate_model(
            model, load_val_data, len(df_val), loss_fn)

        print(f"-- Val   loss {val_loss} accuracy {val_acc}")
        print()

        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), "out/best_model_state.bin")
            best_accuracy = val_acc

    return history, best_accuracy, model, loss_fn, optimizer


def predict(
    df, config, model=None, load_from_path=None, has_labels=False, device=DEVICE
):
    if not model and load_from_path:
        model = SentimentClassifier(
            len(config["CLASS_NAMES"]), config["PRE_TRAINED_MODEL_NAME"]
        )
        model.load_state_dict(torch.load(load_from_path))
        model = model.to(device)

    model = model.eval()
    y_pred = []

    tokenizer = BertTokenizer.from_pretrained(config["PRE_TRAINED_MODEL_NAME"])

    if has_labels:
        y_true = []
    else:
        df["sentiment"] = np.ones(len(df))

    load_test_data = load_data(
        df,
        tokenizer,
        config["MAX_LEN"],
        config["BATCH_SIZE"],
        config["Y_LABEL"],
        config["X_LABEL"],
    )

    with torch.no_grad():
        for d in load_test_data:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)

            if has_labels:
                labels = d["labels"].to(device)
                y_true.extend(labels)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, prediction = torch.max(outputs, dim=1)

            y_pred.extend(prediction)

    y_pred = torch.stack(y_pred).cpu()

    if has_labels:
        y_true = torch.stack(y_true).cpu()
        return y_pred, y_true
    else:
        return y_pred
