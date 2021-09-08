import torchvision.models as models
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class FitNet_4(nn.Module):
    default_input_size = 32

    def __init__(self, in_channels=3, out_channels=10):
        super(FitNet_4, self).__init__()
        self.c1_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.c1_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.c1_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.c1_4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(48),
            nn.ReLU(inplace=True))
        self.c1_5 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(48),
            nn.ReLU(inplace=True))
        self.c2_1 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=80, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(80),
            nn.ReLU(inplace=True))
        self.c2_2 = nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=80, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(80),
            nn.ReLU(inplace=True))
        self.c2_3 = nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=80, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(80),
            nn.ReLU(inplace=True))
        self.c2_4 = nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=80, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(80),
            nn.ReLU(inplace=True))
        self.c2_5 = nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=80, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(80),
            nn.ReLU(inplace=True))
        self.c3_1 = nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=80, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(80),
            nn.ReLU(inplace=True))
        self.c3_2 = nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=80, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(80),
            nn.ReLU(inplace=True))
        self.c3_3 = nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=80, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(80),
            nn.ReLU(inplace=True))
        self.c3_4 = nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=80, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(80),
            nn.ReLU(inplace=True))
        self.c3_5 = nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=80, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(80),
            nn.ReLU(inplace=True))

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout2d(p=0.5)
        fc_infeatures = int(((self.default_input_size / 2 / 2 / 2) ** 2)) * 80
        self.FC = nn.Linear(in_features=fc_infeatures, out_features=out_channels)

    def forward(self, input):
        x = self.c1_1(input)
        x = self.c1_2(x)
        x = self.c1_3(x)
        x = self.c1_4(x)
        x = self.c1_5(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.c2_1(x)
        x = self.c2_2(x)
        x = self.c2_3(x)
        x = self.c2_4(x)
        x = self.c2_5(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.c3_1(x)
        x = self.c3_2(x)
        x = self.c3_3(x)
        x = self.c3_4(x)
        x = self.c3_5(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.FC(torch.flatten(x, 1))
        return x
