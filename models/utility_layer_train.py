import torchvision.models as models
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class DM_pignistic(nn.Module):
    def __init__(self, num_class):
        super(DM_pignistic, self).__init__()
        self.num_class = num_class

    def forward(self, inputs):  # [BS,11]
        avg_Pignistic = inputs[:, -1] / self.num_class
        avg_Pignistic = avg_Pignistic.unsqueeze(-1)
        Pignistic_prob = inputs[:, :] + avg_Pignistic
        Pignistic_prob = Pignistic_prob[:, 0:-1]
        return Pignistic_prob


class DM(nn.Module):
    def __init__(self, nu, num_class):
        super(DM, self).__init__()
        self.nu=nu
        self.num_class=num_class

    def forward(self,inputs):       #[BS,11]
        upper=(1-self.nu)*inputs[:,-1]
        upper=upper.unsqueeze(-1)
        upper=upper.repeat(1,self.num_class+1)
        outputs=inputs+upper
        outputs=outputs[:,0:-1]
        return outputs
