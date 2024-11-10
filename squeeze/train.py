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
    wandb.login(key=args.wandb_key)
    wandb.init(project=args.project_name,
               name=args.model+'_lr_'+str(args.lr_teacher)+'_bs_'+str(args.batch_size))
    experiment_table = wandb.Table(columns=['Model_name', "Val Accuracy","Train Accuracy",'Learning_rate','Batch_size','Time_to_train','Model_parameters'])
    args.device = 'cuda:'+args.gpu_device if torch.cuda.is_available() else 'cpu'
    wandb.run.name = args.model+'_lr_'+str(args.lr_teacher)+'_bs_'+str(args.batch_size)
    wandb.run.save()
    
    ''' Training_dataset'''
    train_set, val_set = load_data(args.data_path,args.data_path, args)
    
    ''' Model Selection and Model path'''
    
    assert args.model in ["ResNet50","ResNet18", "MobileNetV2", "ShuffleNetV2_0_5", "WRN_16_2",
                          "ConvNetW128","efficientnet_b0","efficientnet_v2_s"], f"{args.model} must be one of ResNet18, MobileNetV2, ShuffleNetV2_0_5, WRN_16_2!"
    
    if args.model in ["ResNet18", "MobileNetV2", "ShuffleNetV2_0_5","ResNet50"]:
        model = model_dict[args.model](num_classes=args.num_classes)
    elif args.model =="efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
# Optional: Modify the classifier layer for a specific number of classes
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, args.num_classes)

    model.to(args.device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('\n================== Exp %d ==================\n '%exp)
    print('Hyper-parameters: \n', args.__dict__)
    save_dir = os.path.join(args.squeeze_path, args.dataset, args.model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ''' Loss and scheduler '''
    criterion = nn.CrossEntropyLoss().to(args.device)
    model = model.to(args.device)

    
    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr_teacher,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters, lr=args.lr_teacher, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters, lr=args.lr_teacher,weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")
    

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.train_epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )
    

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler
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
            input = input.float().to(args.device)
            target =  target.to(args.device)
            optimizer.zero_grad()
            logit = model(input)
            loss = criterion(logit, target)
            loss.backward()
            optimizer.step()
            total_acc += (target == logit.argmax(1)).float().sum().item()
            total_number += target.shape[0]

        lr_scheduler.step()

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
                    output=model(input.float().to(args.device))
                    target= torch.tensor(target).to(args.device)
                    total_acc += (target == output.argmax(1)).float().sum().item()
                    total_number += target.shape[0]
                top_1_acc_val = round(total_acc * 100 / total_number, 3)
                print(f"Epoch: {e}, Top-1 Validation Accuracy: {top_1_acc_val}%")
                wandb.log({"epoch": e,"Training_val": top_1_acc_val})
            model.train()
    t2=time.time()
    experiment_table.add_data(args.model, top_1_acc_val, top_1_acc_train,args.lr_teacher,args.batch_size,t2-t1,trainable_params)
    wandb.log({"experiment_results": experiment_table})
    wandb.finish()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='Tiny-ImageNet', help='dataset')
    parser.add_argument('--project_name', type=str, default='EDC_squeeze_final', help='project_name')
    parser.add_argument('--gpu-device', type=str, default='0', help='gpu_device')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--lr-teacher', type=float, default=0.1, help='learning rate for updating network parameters')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size for training networks')
    parser.add_argument('--num-classes', type=int, default=200, help='Number of classes in the dataset')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--squeeze_path', type=str, default='./squeeze_wo_ema/', help='squeeze path')
    parser.add_argument('--train-epochs', type=int, default=51)
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularization')
    parser.add_argument("--wandb-key", default=None, type=str, help="the key for wandb login")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--wd","--weight-decay",default=1e-4,type=float,metavar="W",help="weight decay (default: 1e-4)",dest="weight_decay")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    args = parser.parse_args()
    main(args)
