import torch

from utils import data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


