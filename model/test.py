# Functions
from dataset import GTSRB
import torch
from model import GTSRB_MODEL

from tvt_transforms import transforms

from utils import load_model

import tqdm
from device import device
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

INPUT_DIM = 3 * 50 * 50
OUTPUT_DIM = 43
LEARNING_RATE = 0.0008

test_dataset = GTSRB(root='/home/amber/Documents/sign/gtsrb-german-traffic-sign', split='test', transform=transforms)
test_dataloader = DataLoader(dataset=test_dataset)

# Loading model
filepath = f'/home/amber/Documents/sign/model/checkpoint.pkl'
trained_model = load_model(filepath, INPUT_DIM, OUTPUT_DIM, LEARNING_RATE)
trained_model.eval()

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
