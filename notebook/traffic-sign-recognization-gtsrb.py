#!/usr/bin/env python
# coding: utf-8
import opendatasets as od

od.download("https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")

import numpy as np
import pandas as pd

from model.utils import train_test_split
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from torch.optim import Adam, lr_scheduler

np.random.seed(42)
from matplotlib import style
style.use('fivethirtyeight')

# Custom dataset
from model.dataset import GTSRB
# import transforms methods to train/validation/test data
from model.tvt_transforms import train_transforms, validation_transforms, transforms

# Read data
data_dir = '/home/amber/Documents/sign/gtsrb-german-traffic-sign'
train_data = GTSRB(root=data_dir, split="train")
test_data = GTSRB(root=data_dir, split="test")

train_dataset = GTSRB(root='/home/amber/Documents/sign/gtsrb-german-traffic-sign', split="train")
train_set, validation_set = train_test_split(train_dataset, train_size=0.8)

# ### Transformation

# In[28]:


train_set.dataset.transform = train_transforms
validation_set.dataset.transform = validation_transforms

# ### Dataloader

# In[29]:


BATCH_SIZE = 64
train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(dataset=validation_set, batch_size=BATCH_SIZE)

# ### Modeling

# In[30]:


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running on the GPU")
else:
    device = torch.device("cpu")
    print("running on the CPU")

class GTSRB_MODEL(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GTSRB_MODEL, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.metrics = {}

        self.flatten = nn.Flatten()

        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3)
        self.batchnorm3 = nn.BatchNorm2d(1024)

        self.l1 = nn.Linear(1024 * 4 * 4, 512)
        self.l2 = nn.Linear(512, 128)
        self.batchnorm4 = nn.LayerNorm(128)
        self.l3 = nn.Linear(128, output_dim)

    def forward(self, input):

        conv = self.conv1(input)
        conv = self.conv2(conv)
        batchnorm = self.relu(self.batchnorm1(conv))
        maxpool = self.maxpool(batchnorm)

        conv = self.conv3(maxpool)
        conv = self.conv4(conv)
        batchnorm = self.relu(self.batchnorm2(conv))
        maxpool = self.maxpool(batchnorm)

        conv = self.conv5(maxpool)
        conv = self.conv6(conv)
        batchnorm = self.relu(self.batchnorm3(conv))
        maxpool = self.maxpool(batchnorm)

        flatten = self.flatten(maxpool)

        dense_l1 = self.l1(flatten)
        dropout = self.dropout3(dense_l1)
        dense_l2 = self.l2(dropout)
        batchnorm = self.batchnorm4(dense_l2)
        dropout = self.dropout2(batchnorm)
        output = self.l3(dropout)

        return output

    def training_metrics(self, positives, data_size, loss):
        acc = positives / data_size
        return loss, acc

    def validation_metrics(self, validation_data, loss_function):
        data_size = len(validation_data)
        correct_predictions = 0
        total_samples = 0
        val_loss = 0

        model = self.eval()
        with torch.no_grad():
            for step, (input, label) in enumerate(validation_data):
                input = input.to(torch.float)

                input, label = input.to(device), label.to(device)
                prediction = model.forward(input)
                loss = loss_function(prediction, label)
                val_loss = loss.item()
                _, predicted = torch.max(prediction, 1)
                correct_predictions += (predicted == label).sum().item()
                total_samples += label.size(0)

        val_acc = correct_predictions / total_samples

        return val_loss, val_acc

    def history(self):
        return self.metrics

    def compile(self, train_data, validation_data, epochs, loss_function, optimizer, learning_rate_scheduler):
        val_acc_list = []
        val_loss_list = []

        train_acc_list = []
        train_loss_list = []

        learning_rate_list = []

        print('training started ...')
        STEPS = len(train_data)
        for epoch in range(epochs):
            lr = optimizer.param_groups[0]["lr"]
            learning_rate_list.append(lr)
            correct_predictions = 0
            total_examples = 0
            loss = 0
            with tqdm.trange(STEPS) as progress:

                for step, (input, label) in enumerate(train_loader):
                    input = input.to(torch.float)
                    #                     input.type()

                    input, label = input.to(device), label.to(device)
                    prediction = self.forward(input)

                    _, predicted = torch.max(prediction, 1)
                    correct_predictions += (predicted == label).sum().item()
                    total_examples += label.size(0)
                    l = loss_function(prediction, label)
                    loss = l.item()
                    l.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    progress.colour = 'green'
                    progress.desc = f'Epoch [{epoch}/{EPOCHS}], Step [{step}/{STEPS}], Learning Rate [{lr}], Loss [{"{:.4f}".format(l)}], Accuracy [{"{:.4f}".format(correct_predictions / total_examples)}]'
                    progress.update(1)

            training_loss, training_acc = self.training_metrics(correct_predictions, total_examples, loss)
            train_acc_list.append(training_acc)
            train_loss_list.append(training_loss)

            val_loss, val_acc = self.validation_metrics(validation_data, loss_function)
            val_acc_list.append(val_acc)
            val_loss_list.append(val_loss)

            print(f'val_accuracy [{val_acc}], val_loss [{val_loss}]')

            learning_rate_scheduler.step()

        metrics_dict = {
            'train_acc': train_acc_list,
            'train_loss': train_loss_list,
            'val_acc': val_acc_list,
            'val_loss': val_loss_list,
            'learning_rate': optimizer.param_groups[0]["lr"]
        }
        self.metrics = metrics_dict
        print('training complete !')


import tqdm

EPOCHS = 10
LEARNING_RATE = 0.0008
INPUT_DIM = 3 * 50 * 50
OUTPUT_DIM = 43
model = GTSRB_MODEL(INPUT_DIM, OUTPUT_DIM).to(device)

optimizer = Adam(params=model.parameters(), lr=LEARNING_RATE)
lr_s = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=10)
loss = nn.CrossEntropyLoss()

model.compile(train_data=train_loader,
              validation_data=validation_loader,
              epochs=EPOCHS,
              loss_function=loss,
              optimizer=optimizer,
              learning_rate_scheduler=lr_s)

# saving a checkpoint assuming the network class named ClassNet
checkpoint = {'model': model,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'epoch': EPOCHS,
              'loss': loss, }

torch.save(checkpoint, 'checkpoint.pkl')

# Loading model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    model = checkpoint['model']  # 提取网络结构
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器参数
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']

    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()

    return model


trained_model = load_checkpoint('/home/amber/Documents/sign/model/checkpoint.pkl')

test_dataset = GTSRB(root='/home/amber/Documents/sign/gtsrb-german-traffic-sign', split='test', transform=transforms)
test_dataloader = DataLoader(dataset=test_dataset)
print(test_dataloader)

from sklearn.metrics import accuracy_score

y_pred = []
y_true = []
trained_model = trained_model.eval().to(device)
with tqdm.tqdm(colour='red', total=len(test_dataloader)) as progress:
    with torch.no_grad():
        for id, (input, label) in enumerate(iter(test_dataloader)):
            input = input.to(torch.float)
            input, label = input.to(device), label
            y_true = label.cpu().detach().numpy()

            prediction = trained_model.forward(input)
            _, prediction = torch.max(prediction, 1)
            y_pred = prediction.cpu().detach().numpy()

            progress.desc = f'Test Accuracy : {accuracy_score(y_true, y_pred)} '
            progress.update(1)
