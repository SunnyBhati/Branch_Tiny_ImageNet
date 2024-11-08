import os,math
import argparse
import sys

import torch.utils
sys.path.append("../")
import torch
import time
import torchvision
from torchvision import transforms
import copy
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from baseline import get_network as ti_get_network
from torchvision import models
from models import model_dict
from tiny_in_dataset_new import TinyImageNet
import wandb

def load_data(traindir, valdir, args):
    # Data loading code
    normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                     std=[0.2302, 0.2265, 0.2262])
    print("Loading training data")
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    dataset = TinyImageNet(traindir, split='train', download=True, transform=train_transform)
    print("Loading validation data")
    val_transform = transforms.Compose([transforms.ToTensor(),normalize])
    dataset_test = TinyImageNet(valdir, split='val', download=True, transform=val_transform)
    return dataset, dataset_test

def main(args):
    ''' wand initialization'''
    wandb.login(key="085788de93539cf40689d3714b6dc50a54f20a86")
    wandb.init(project=args.project_name,name=args.model+'_lr_'+str(args.lr_teacher)+'_bs_'+str(args.batch_size))
    experiment_table = wandb.Table(columns=['Model_name', "Val Accuracy","Train Accuracy",'Learning_rate','Batch_size','Time_to_train'])
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    train_set, val_set = load_data(args.data_path,args.data_path, args)
    
    ''' Model Selection and Model path'''
    
    assert args.model in ["ResNet18", "MobileNetV2", "ShuffleNetV2_0_5", "WRN_16_2",
                          "ConvNetW128","efficientnet_b0","efficientnet_v2_s"], f"{args.model} must be one of ResNet18, MobileNetV2, ShuffleNetV2_0_5, WRN_16_2!"
    if args.model == "ConvNetW128":
        model = ti_get_network(args.model, channel=3, num_classes=200, im_size=(64, 64), dist=False)
    elif args.model == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
# Optional: Modify the classifier layer for a specific number of classes
        num_classes = 200  # Replace with the number of classes for your task
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    elif args.model == "efficientnet_v2_s":
        model = models.efficientnet_b0(weights=None)
        num_classes = 200  # Replace with the number of classes for your task
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        model = model_dict[args.model](num_classes=200)
    # print('\n================== Exp %d ==================\n '%exp)
    print('Hyper-parameters: \n', args.__dict__)
    save_dir = os.path.join(args.squeeze_path, args.dataset, args.model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ''' Loss and scheduler '''
    criterion = nn.CrossEntropyLoss().to(args.device)
    model = model.to(args.device)
    lr = args.lr_teacher
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lambda step: 0.5 * (
                                                          1. + math.cos(
                                                      math.pi * step / args.train_epochs)) if step <= args.train_epochs else 0,
                                                  last_epoch=-1)
    ''' organize the real dataset '''
    trainloader = torch.utils.data.DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4)
    
    val_loader= torch.utils.data.DataLoader(val_set,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=4)
    model.train()

    ''' Training Loop'''
    t1=time.time()
    for e in range(args.train_epochs):
        total_acc = 0
        total_number = 0
        model.train()

        for batch_idx, (input, target) in enumerate(trainloader):
            input = input.float().cuda()
            target =  target.cuda()
            optimizer.zero_grad()
            logit = model(input)
            loss = criterion(logit, target)
            loss.backward()
            optimizer.step()

            total_acc += (target == logit.argmax(1)).float().sum().item()
            total_number += target.shape[0]

        scheduler.step()

        top_1_acc_train = round(total_acc * 100 / total_number, 3)
        print(f"Epoch: {e}, Top-1 Accuracy: {top_1_acc_train}%")
       
        if e % 10 == 0:
            wandb.log({"epoch": e, "loss_train": loss,"Training_acc": top_1_acc_train})
            state_dict = model.state_dict()
            torch.save(state_dict, os.path.join(save_dir, f"squeeze_{args.model}.pth"))

        if e % 10 == 0:
            model.eval()
            total_acc = 0
            total_number = 0
            with torch.no_grad():
                for batch_idx, (input, target) in enumerate(val_loader):
                    output=model(input.float().cuda())
                    target= torch.tensor(target).cuda()
                    total_acc += (target == output.argmax(1)).float().sum().item()
                    total_number += target.shape[0]
                top_1_acc_val = round(total_acc * 100 / total_number, 3)
                print(f"Epoch: {e}, Top-1 Validation Accuracy: {top_1_acc_val}%")
                wandb.log({"epoch": e,"Training_val": top_1_acc_val})
            model.train()
    t2=time.time()
    experiment_table.add_data(args.model, top_1_acc_val, top_1_acc_train,args.lr_teacher,args.batch_size,t2-t1)
    wandb.log({"experiment_results": experiment_table})
    wandb.finish()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='Tiny-ImageNet', help='dataset')
    parser.add_argument('--project_name', type=str, default='EDC_squeeze', help='project_name')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--lr_teacher', type=float, default=0.1, help='learning rate for updating network parameters')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training networks')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--squeeze_path', type=str, default='./squeeze_wo_ema/', help='squeeze path')
    parser.add_argument('--train_epochs', type=int, default=51)
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularization')
    args = parser.parse_args()
    main(args)
