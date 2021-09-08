import torch
import torchvision
import torchvision.transforms as transforms
import argparse
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score, classification_report
from models import FitNet_4
from torch import optim


def train(arg):
    ##########basic params############
    epochs = 2000
    expname="exp1"
    #########datasets##################
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)
    pbar_train = tqdm(trainloader)
    pbar_test = tqdm(testloader)
    ###################################
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    ############model#################
    model = FitNet_4.FitNet_4(in_channels=3, out_channels=10)
    model = nn.DataParallel(model)
    model.cuda()
    ############loss##################
    criterion = nn.CrossEntropyLoss()
    ############optimizer############
    optimizer = optim.Adam()
    for epoch in range(1,epochs+1):
        train_loss = 0
        train_acc = 0
        model.train()
        for (image, label) in pbar_train:
            image = Variable(image).cuda()
            label = Variable(label).cuda()
            out = model(image)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar_train.set_description("Training:epoch{} loss:{}".format(epoch, loss.item()))
            train_loss += loss.item() * label.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == label).sum()
            train_acc += num_correct.item()
        print("Training{}:epoch{} finished! Total acc:{} Total loss:{}".format(expname, epoch,
                                                                               train_acc / len(trainset),
                                                                               train_loss / (len(trainset))))


if __name__ == '__main__':
    train(1)
