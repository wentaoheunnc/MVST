import os
import time
import sys
from copy import deepcopy

import random
import math
import torch.backends.cudnn as cudnn

import numpy as np

import torch
from torch.utils.data import Dataset
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from gated_fusion import new_fuse
from util.icbhi_util import get_score


def load_feature(path):
    file_list = os.listdir(path)
    file_list.sort()
    num_batches = len(file_list) // 2
    featr_list = file_list[:num_batches]
    label_list = file_list[num_batches:]

    features = []
    for bch in featr_list:
        featr = torch.load(os.path.join(path,bch))
        features.append(featr)
    features = torch.cat(features, dim=0)

    labels = []
    for bch in label_list:
        label = torch.load(os.path.join(path,bch))
        labels.append(label)
    labels = torch.cat(labels, dim=0)

    return features, labels


class MyDataset(Dataset):
    def __init__(self, features1, features2, features3, features4, features5, labels, transform=None):
        self.features1 = features1
        self.features2 = features2
        self.features3 = features3
        self.features4 = features4
        self.features5 = features5
        self.labels = labels
        self.transform = transform

        self.class_nums = np.zeros(4)
        for sample in self.labels:
            self.class_nums[sample] += 1
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        feature1 = self.features1[index]
        feature2 = self.features2[index]
        feature3 = self.features3[index]
        feature4 = self.features4[index]
        feature5 = self.features5[index]
        label = self.labels[index]

        return feature1, feature2, feature3, feature4, feature5, label

class AverageMeter(object):
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        n_cls = output.shape[1]
        valid_topk = [k for k in topk if k <= n_cls]
        
        maxk = max(valid_topk)
        bsz = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            if k in valid_topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / bsz))
            else: res.append(torch.tensor([0.]))

        return res, bsz

def train(train_loader, model, loss_function, optimizer, epoch):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        data_time.update(time.time() - end)

        feature1, feature2, feature3, feature4, feature5, labels = data
        feature1 = feature1.cuda()
        feature2 = feature2.cuda()
        feature3 = feature3.cuda()
        feature4 = feature4.cuda()
        feature5 = feature5.cuda()
        labels = labels.cuda()

        bsz = labels.shape[0]

        with torch.cuda.amp.autocast():
            output = model(feature1, feature2, feature3, feature4, feature5)
            loss = loss_function(output, labels)

        losses.update(loss.item(), bsz)
        [acc1], _ = accuracy(output[:bsz], labels, topk=(1,))
        top1.update(acc1[0], bsz)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % 10 == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg

def validate(val_loader, model, loss_function, best_acc, best_model=None):
    save_bool = False
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    hits, counts = [0.0] * 4, [0.0] * 4

    with torch.no_grad():
        end = time.time()
        for idx, data in enumerate(val_loader):
            feature1, feature2, feature3, feature4, feature5, labels = data
            feature1 = feature1.cuda()
            feature2 = feature2.cuda()
            feature3 = feature3.cuda()
            feature4 = feature4.cuda()
            feature5 = feature5.cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            with torch.cuda.amp.autocast():
                output = model(feature1, feature2, feature3, feature4, feature5)
                loss = loss_function(output, labels)

            losses.update(loss.item(), bsz)
            [acc1], _ = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0], bsz)

            _, preds = torch.max(output, 1)
            for idx in range(preds.shape[0]):
                counts[labels[idx].item()] += 1.0
                
                if preds[idx].item() == labels[idx].item():
                    hits[labels[idx].item()] += 1.0

            sp, se, sc = get_score(hits, counts)

            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % 10 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx + 1, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))
    
    if sc > best_acc[-1] and se > 5:
        save_bool = True
        best_acc = [sp, se, sc]
        best_model = [deepcopy(model.state_dict())]

    print(' * S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f} (Best S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f})'.format(sp, se, sc, best_acc[0], best_acc[1], best_acc[-1]))
    print(' * Acc@1 {top1.avg:.2f}'.format(top1=top1))

    return best_acc, best_model, save_bool


def main():

    # fix seed
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    featr_train_16, label_train_16 = load_feature("./features_train/16")
    featr_test_16, label_test_16 = load_feature("./features_test/16")

    featr_train_32, label_train_32 = load_feature("./features_train/32")
    featr_test_32, label_test_32 = load_feature("./features_test/32")

    featr_train_64, label_train_64 = load_feature("./features_train/64")
    featr_test_64, label_test_64 = load_feature("./features_test/64")

    featr_train_128, label_train_128 = load_feature("./features_train/128")
    featr_test_128, label_test_128 = load_feature("./features_test/128")

    featr_train_256, label_train_256 = load_feature("./features_train/256")
    featr_test_256, label_test_256 = load_feature("./features_test/256")

    train_dataset = MyDataset(featr_train_16, featr_train_32, featr_train_64, featr_train_128, featr_train_256, label_train_16)
    test_dataset = MyDataset(featr_test_16, featr_test_32, featr_test_64, featr_test_128, featr_test_256, label_test_16)

    weights = torch.tensor(train_dataset.class_nums, dtype=torch.float32)
    weights = (weights / weights.sum())
    loss_function_train = torch.nn.CrossEntropyLoss(weight=weights)
    loss_function_train = loss_function_train.cuda()

    weights = torch.tensor(test_dataset.class_nums, dtype=torch.float32)
    weights = (weights / weights.sum())
    loss_function_test = torch.nn.CrossEntropyLoss(weight=weights)
    loss_function_test = loss_function_test.cuda()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True,
                                               pin_memory=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False,
                                              pin_memory=False, num_workers=0)

    fusion = new_fuse(num_features=5, num_classes=4).cuda()
    optimizer = optim.AdamW(fusion.parameters(), lr=0.001, weight_decay=1E-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-15)

    epochs = 100
    best_acc = [0, 0, 0]
    best_model = None

    for epoch in range(epochs):
        time1 = time.time()
        loss, acc = train(train_loader, fusion, loss_function_train, optimizer, epoch)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2-time1, acc))

        scheduler.step()

        best_acc, best_model, save_bool = validate(test_loader, fusion, loss_function_train, best_acc, best_model)


            

main()