from torch.optim import Adam
import torch
from model import GTSRB_MODEL
from device import device
def load_model(filepath, INPUT_DIM, OUTPUT_DIM, LEARNING_RATE):
    checkpoint = torch.load(filepath)
    model = GTSRB_MODEL(INPUT_DIM, OUTPUT_DIM).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    optimizer = Adam(params=model.parameters(), lr=LEARNING_RATE)

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器参数

    for parameter in model.parameters():
        parameter.requires_grad = False
    return model
