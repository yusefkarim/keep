# Functions
import csv
import pathlib
from typing import Tuple, Optional, Callable

import PIL
import torch
import torch.nn as nn
import tqdm
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running on the GPU")
else:
    device = torch.device("cpu")
    print("running on the CPU")


class GTSRB(Dataset):
    def __init__(self,
                 root: str,
                 split: str,
                 transform: Optional[Callable] = None):
        self.base_folder = pathlib.Path(root)
        self.csv_file = self.base_folder / ('Train.csv' if split == 'train' else 'Test.csv')

        with open('/' + str(self.csv_file)) as csvfile:
            samples = [('/' + str(self.base_folder / row['Path']), int(row['ClassId']))
                       for row in csv.DictReader(csvfile, delimiter=',', skipinitialspace=True)
                       ]

        self.samples = samples
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple:
        path, classId = self.samples[index]
        sample = PIL.Image.open(path).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, classId


# Net
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

    # Transforms


transforms = v2.Compose([
    v2.Resize(size=(50, 50)),
    v2.ToImageTensor(),

])

test_dataset = GTSRB(root='/home/amber/Documents/sign/gtsrb-german-traffic-sign', split='test', transform=transforms)
test_dataloader = DataLoader(dataset=test_dataset)

INPUT_DIM = 3 * 50 * 50
OUTPUT_DIM = 43
LEARNING_RATE = 0.0008

# Loading model
trained_model = GTSRB_MODEL(INPUT_DIM, OUTPUT_DIM).to(device)

filepath = f'/home/amber/Documents/sign/model/checkpoint.pkl'

checkpoint = torch.load(filepath)

trained_model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
optimizer = Adam(params=trained_model.parameters(), lr=LEARNING_RATE)

optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器参数

for parameter in trained_model.parameters():
    parameter.requires_grad = False
trained_model.eval()

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
