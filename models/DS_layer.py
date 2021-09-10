import torchvision.models as models
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

"""
from the paper
①   In the ENN classifier, the proximity of an input vector to prototypes is considered as evidence about its class. 
   This evidence is converted into mass functions and combined using Dempster’s rule.
"""


class DS1(nn.Module):
    def __init__(self, units, input_dim):
        super(DS1, self).__init__()
        self.w = nn.Parameter(torch.zeros(units, input_dim))  # self.w尺寸：[200,128] 生成了200个prototype，每一个有input_dim维。
        self.units = units
        nn.init.kaiming_normal(self.w)

    def forward(self, inputs):  # 输入的尺寸[BS,128]
        for i in range(self.units):
            if i == 0:
                un_mass_i = self.w[i, :] - inputs           #self.w[i,:]的尺寸（128,)  un_mass_i的尺寸(1,128)
                un_mass_i = torch.square(un_mass_i)
                un_mass_i = torch.sum(un_mass_i, dim=-1, keepdim=True)      #求和之后un_mass_i尺寸(1,1),这个结果就是input和第i个prototype的相似度
                un_mass = un_mass_i
            if i >= 1:
                un_mass_i = self.w[i, :] - inputs
                un_mass_i = torch.square(un_mass_i)
                un_mass_i = torch.sum(un_mass_i, dim=-1, keepdim=True)
                un_mass = torch.cat([un_mass, un_mass_i], -1)

        return un_mass  # [B,200]   计算出的结果是这一组input和200个prototype的相似度（距离）


class DS1_activate(nn.Module):
    """
    This Module could refer to the section2.2 in the Paper.
    The distance-based support between x and each reference pattern pi.
    Here the inputs are Euclidean Distance (not after sqrt) between deep features and prototypes.
    """
    def __init__(self, input_dim):      #input_dim这里接收的是外面的self.prototype,即prototype的数量
        super(DS1_activate, self).__init__()
        self.x1 = nn.Parameter(torch.zeros(1, input_dim))           #[1,128]
        self.eta = nn.Parameter(torch.zeros(1, input_dim))          #[1,128]
        self.input_dim = input_dim
        nn.init.kaiming_normal(self.x1)
        nn.init.kaiming_normal(self.eta)

    def forward(self, inputs):
        """this part corresponds to the formula 5 in the Paper"""
        gamma = torch.square(self.eta)  #[1,200]. Note that the inputs here are of square format already, thus there is no need to do redundant squaring for inputs.
        alpha = torch.negative(self.x1) #[1,200]
        alpha = torch.exp(alpha)+1
        alpha = torch.div(1, alpha)         #这一步保证了alpha在（0,1)之间
        si = gamma * inputs
        si = torch.negative(si)
        si = torch.exp(si)
        si = si * alpha
        si = si / (torch.max(si, dim=-1, keepdim=True)[0] + 0.0001)             #在这里归一化很重要
        return si


class DS2(nn.Module):
    def __init__(self, input_dim, num_class):  # 200,10
        super(DS2, self).__init__()
        self.beta = nn.Parameter(torch.zeros(input_dim, num_class))     #[200,10]
        self.input_dim = input_dim
        self.num_class = num_class
        nn.init.kaiming_normal(self.beta)
    def forward(self, inputs):  # [bs,200]
        beta = torch.square(self.beta)  # (200,10)
        beta_sum = torch.sum(beta, dim=-1, keepdim=True)  # (200,1）
        u = torch.div(beta, beta_sum)  # (200,10), 这里除以和很重要，保证了每一个Ui（shape(10,))中所有元素相加等于0
        inputs_new = torch.unsqueeze(inputs, -1)  # [bs,200,1]
        for i in range(self.input_dim):
            if i == 0:
                mass_prototype_i = u[i, :] * inputs_new[:, i]           #u[i, :]-shape-[10,],  inputs_new[:, i]-shape-[bs,1],这一步计算出的是和10个类别相关的东西
                mass_prototype = mass_prototype_i.unsqueeze(-2)
            if i >= 1:
                mass_prototype_i = u[i, :] * inputs_new[:, i]
                mass_prototype_i = mass_prototype_i.unsqueeze(-2)
                mass_prototype = torch.cat([mass_prototype, mass_prototype_i], -2)
        # u每一个分量之和都在0-1之间, inputs也在DS activate中被归一化到了0-1，所以这相乘得到的结果mass_prototype也在0-1之间
        return mass_prototype


class DS2_omega(nn.Module):
    def __init__(self, input_dim, num_class):
        super(DS2_omega, self).__init__()
        self.input_dim = input_dim
        self.num_class = num_class

    def forward(self, inputs):  # [bs,200,10]
        mass_omega_sum = torch.sum(inputs, dim=-1, keepdim=True)
        mass_omega_sum = 1 - mass_omega_sum[:, :, 0]
        mass_omega_sum = mass_omega_sum.unsqueeze(-1)
        mass_with_omega = torch.cat([inputs, mass_omega_sum], -1)
        return mass_with_omega


class DS3_Dempster(nn.Module):
    def __init__(self, input_dim, num_class):
        super(DS3_Dempster, self).__init__()
        self.input_dim=input_dim
        self.num_class=num_class


    def forward(self,inputs):   #[Bs,200,11]
        m1=inputs[:,0,:]
        omega1=torch.unsqueeze(inputs[:,0,-1],-1)
        for i in range(self.input_dim-1):
            m2=inputs[:,(i+1),:]
            omega2=inputs[:,(i+1),-1]
            omega2=omega2.unsqueeze(-1)
            combine1=m1*m2
            combine2=m1*omega2
            combine3=omega1*m2
            combine1_2=combine1+combine2
            combine2_3=combine1_2+combine3
            combine2_3=combine2_3/torch.sum(combine2_3,dim=-1,keepdim=True)
            m1=combine2_3
            omega1=torch.unsqueeze(combine2_3[:,-1],-1)

        return m1

class DS3_normalize(nn.Module):
    def __init__(self):
        super(DS3_normalize, self).__init__()

    def forward(self,inputs):           #[bs,11]
        mass_combine_normalize=inputs/torch.sum(inputs,axis=-1,keepdim=True)
        return mass_combine_normalize


if __name__ == '__main__':
    ds = DS1(200, 128)
    test_input = torch.randn(4, 128)
    ds(test_input)
