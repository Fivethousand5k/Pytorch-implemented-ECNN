import numpy
from sklearn.metrics import cohen_kappa_score, classification_report
import torch
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score, classification_report
from models import FitNet_4
from torch import optim
import numpy as np


def evaluation(test_dataloader, model, class_names, epoch, criterion):
    eval_loss_list = []
    eval_acc = 0
    pred_list = []
    GT_list = []
    pbar_test = tqdm(test_dataloader, total=len(test_dataloader))
    with torch.no_grad():
        for image, label in pbar_test:
            image = Variable(image).cuda()
            label = Variable(label).cuda()
            out = model(image)
            loss = criterion(out, label)
            eval_loss_list.append(loss.item())
            _, pred = torch.max(out, 1)
            num_correct = (pred == label).sum()
            pred_list.extend(pred.cpu().numpy().tolist())
            GT_list.extend(label.cpu().numpy().tolist())
            eval_acc += num_correct.item()
            pbar_test.set_description("Testing:epoch{} loss:{}".format(epoch, loss.item()))
    epoch_test_acc = eval_acc / len(pbar_test)
    print(
        "Testing:epoch{} finished! Total loss:{}".format(epoch,
                                                                      np.mean(eval_loss_list)))
    print(classification_report(y_true=GT_list, y_pred=pred_list, target_names=class_names))
    kappa = cohen_kappa_score(y1=pred_list, y2=GT_list)
    print("Kappa:{}".format(kappa))
