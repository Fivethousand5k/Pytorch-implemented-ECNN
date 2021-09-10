import torchvision.models as models
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from models.DS_layer import *
from models.utility_layer_train import *


class DS_FitNet_4(nn.Module):
    default_input_size = 32
    prototype = 200

    def __init__(self, in_channels=3, out_channels=10):
        super(DS_FitNet_4, self).__init__()
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
            nn.Conv2d(in_channels=80, out_channels=128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.c3_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.c3_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.c3_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.c3_5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))

        self.DS1 = DS1(self.prototype, 128)
        self.DS1_activate = DS1_activate(self.prototype)
        self.DS2 = DS2(self.prototype, num_class=out_channels)
        self.DS2_omega = DS2_omega(self.prototype, num_class=out_channels)
        self.DS3_Dempster = DS3_Dempster(self.prototype, num_class=out_channels)
        self.DS3_normalize = DS3_normalize()
        self.utility_layer = DM(nu=0.9, num_class=out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.maxpool_final = nn.MaxPool2d(kernel_size=8)
        self.dropout = nn.Dropout2d(p=0.5)
        # fc_infeatures = int(((self.default_input_size / 2 / 2 / 2) ** 2)) * 128
        # self.FC = nn.Linear(in_features=fc_infeatures, out_features=out_channels)

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
        x = self.maxpool_final(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)         #here shape of x: [1,128]
        x = self.DS1(x)
        x = self.DS1_activate(x)
        x = self.DS2(x)
        x = self.DS2_omega(x)
        x = self.DS3_Dempster(x)
        x = self.DS3_normalize(x)
        x = self.utility_layer(x)
        return x


if __name__ == '__main__':
    model = DS_FitNet_4(in_channels=3, out_channels=10)
    test_input = torch.randn(1,3,32,32)
    output=model(test_input)
    print(output)
