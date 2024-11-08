import os
import sys
import math
import time
import shutil
import argparse
import numpy as np
import wandb
import timm
from copy import deepcopy

import torch.distributed as dist
sys.path.append('../')
import relabel.models as ti_models
from baseline import get_network as ti_get_network
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
from torchvision.transforms.functional import to_pil_image
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.optim.lr_scheduler import LambdaLR
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tiny_in_dataset_new import TinyImageNet
from utils import AverageMeter, accuracy, get_parameters
from torch.utils.data._utils.fetch import _MapDatasetFetcher
import torch
import torch.nn as nn
from relabel.utils_fkd import mix_aug
from torch.utils.data import Dataset, DataLoader
torch.cuda.empty_cache()


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
def renormalize_batch(batch):
    """
    Renormalize a batch of images from normalized range to [0, 1] or [0, 255].
    
    Args:
        batch (torch.Tensor): Batch of images with shape [B, C, H, W] in normalized form.

    Returns:
        torch.Tensor: Renormalized batch with values in the [0, 1] range.
    """
    # Mean and std used during normalization (shape adjusted for broadcasting)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    # Unnormalize the batch
    batch = batch * std + mean
    image = batch.clamp(0, 1)
    return image

  
    
    


class CustomDataset(Dataset):
    def __init__(self, full_label, final_images, hard_labels,transform):
        self.full_label = full_label
        self.final_images = final_images
        self.hard_labels = hard_labels
        self.transform = transform

    def __len__(self):
        return len(self.hard_labels)

    def __getitem__(self, idx):
        image = self.final_images[idx]
        label = self.full_label[idx]
        hard_label = self.hard_labels[idx]

        image = (image * 255).clamp(0, 255).to(torch.uint8)  # Ensure valid pixel values
        pil_image = to_pil_image(image)  # Convert to PIL image

        # Apply transformations if provided
        if self.transform:
            pil_image = self.transform(pil_image)

        return  pil_image,label, hard_label

def start_subexperiment(args,name,group_name):
    # Start a W&B run for the subexperiment
    run = wandb.init(
        project=args.wandb_project,  
        name= name,
        reinit=True, 
        group=group_name)  # Optional: Group experiments togethe)
    
    return run

class EMAMODEL(object):
    def __init__(self,model):
        self.ema_model = deepcopy(model)
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)
        self.ema_model.eval()

    @torch.no_grad()
    def ema_step(self,decay_rate=0.999,model=None):
        for param,ema_param in zip(model.parameters(),self.ema_model.parameters()):
            ema_param.data.mul_(decay_rate).add_(param.data, alpha=1. - decay_rate)
    
    @torch.no_grad()
    def ema_swap(self,model=None):
        for param,ema_param in zip(self.ema_model.parameters(),model.parameters()):
            tmp = param.data.detach()
            param.data = ema_param.detach()
            ema_param.data = tmp
    
    @torch.no_grad()
    def __call__(self, pre_z_t,t):
        return self.ema_model.module(pre_z_t)

class ALRS():
    def __init__(self, optimizer, decay_rate=0.95):
        self.optimizer = optimizer
        self.decay_rate = decay_rate
        self.prev_loss = 1e3

    def step(self, now_loss):
        delta = abs(self.prev_loss - now_loss)
        if delta / now_loss < 0.02 and delta < 0.02:
                self.optimizer.param_groups[0]["lr"] *= self.decay_rate
        self.p_lr = p_lr = self.optimizer.param_groups[0]["lr"]
        self.prev_loss = now_loss
        print(f"call auto learning rate scheduler, the learning rate is set as {p_lr}, the current loss is {now_loss}")

    def get_last_lr(self):
        return [self.p_lr]

def load_data(traindir, valdir, args):

    normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                     std=[0.2302, 0.2265, 0.2262])
    train_transform = transforms.Compose([transforms.RandomResizedCrop(64),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize])
    
    dataset = TinyImageNet(traindir, split='train', download=True, transform=train_transform)
    
    val_transform = transforms.Compose([transforms.ToTensor(),
                                        normalize])
    dataset_test = TinyImageNet(valdir, split='val', download=True, transform=val_transform)
    
    return dataset, dataset_test

def get_args():
    parser = argparse.ArgumentParser("FKD Training on ImageNet-1K")
    parser.add_argument('--ipc', type=int, default=10, help='the number of each ipc')
    parser.add_argument('--batch-size', type=int,default=100, help='batch size')
    parser.add_argument('--gradient-accumulation-steps', type=int,default=1, help='gradient accumulation steps for small gpu memory')
    parser.add_argument('--start-epoch', type=int,default=0, help='start epoch')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--epochs', type=int, default=300, help='total epoch')
    parser.add_argument('--sgd-lr', type=float,default=0.05, help='adamw learning rate')
    parser.add_argument('--pre-train-path', type=str,default='../squeeze/squeeze_wo_ema/', help='where to load the pre-trained backbone')
    parser.add_argument('-j', '--workers', default=0, type=int,help='number of data loading workers')
    parser.add_argument('--loss-type', type=str, default="mse_gt", help='the type of the loss function')
    parser.add_argument('--train-dir', type=str, default=None,help='path to training dataset')
    parser.add_argument('--val-dir', type=str,default='/path/to/imagenet/val', help='path to validation dataset')
    parser.add_argument('--output-dir', type=str, default='./save/1024', help='path to output dir')
    parser.add_argument('--ls-type', default="cos",type=str, help='the type of lr scheduler')
    parser.add_argument('--alrs-dr', default=0.9975,type=float, help='the decay rate of ALRS')
    parser.add_argument('--ema-dr', default=0.999,type=float, help='the decay rate of EMA')
    parser.add_argument('--st', default=1.5,type=float, help='the scheduler trick')
    parser.add_argument('--sgd', default=False,action='store_true', help='sgd optimizer')
    parser.add_argument('-lr', '--learning-rate', type=float,default=1.024, help='sgd init learning rate')  # checked
    parser.add_argument('--momentum', type=float,default=0.875, help='sgd momentum')  # checked
    parser.add_argument('--weight-decay', type=float,default=3e-5, help='sgd weight decay')  # checked
    parser.add_argument('--adamw-lr', type=float,default=0.001, help='adamw learning rate')
    parser.add_argument('--cutmix', type=float, default=1.0,help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--adamw-weight-decay', type=float,default=0.01, help='adamw weight decay')
    parser.add_argument('--ce-weight', type=float,default=0.1, help='the weight og cross-entropy loss')
    parser.add_argument('--gpu-id', type=str,default="0,1", help='the id of gpu used')
    parser.add_argument('--model', type=str,default='ResNet18', help='student model name')
    parser.add_argument('--shuffle-patch', default=False, action='store_true',help='if use shuffle-patch')
    parser.add_argument('--keep-topk', type=int, default=200,help='keep topk logits for kd loss')
    parser.add_argument('-T', '--temperature', type=float,default=3.0, help='temperature for distillation loss')
    parser.add_argument('--fkd-path', type=str,default=None, help='path to fkd label')
    parser.add_argument('--wandb-project', type=str,default='EDC_fixed', help='wandb project name')
    parser.add_argument('--wandb-api-key', type=str,default=None, help='wandb api key')
    parser.add_argument('--mix-type', default=None, type=str,choices=['mixup', 'cutmix', None], help='mixup or cutmix or None')
    parser.add_argument('--fkd_seed', default=42, type=int,help='seed for batch loading sampler')
    parser.add_argument('--world-size', default=1, type=int,help='number of nodes for distributed training')
    parser.add_argument('--teacher-name', nargs='+', default=["ResNet18", "ConvNetW128", "MobileNetV2", "WRN_16_2", "ShuffleNetV2_0_5"], help='define the teacher models')
    args = parser.parse_args()
    args.mode = 'fkd_load'

    return args


def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    main_worker(0,args)


def main_worker(gpu,args):
    torch.cuda.set_device(gpu) 

    args.epochs=1000
    
    if args.teacher_name ==["ResNet18", "ConvNetW128" ,"MobileNetV2","WRN_16_2", "ShuffleNetV2_0_5"]:
        model_soft='all'
    else:
        model_soft=args.teacher_name[0]
   
    group_name= args.train_dir.split('/')[-1]
    name_exp= str(time.time())+'Relabel'+'_softlbl_'+str(model_soft)+'_epc_'+str(args.epochs)+'_stumod_'+args.model+'_mixtype_'+args.mix_type


    if not torch.cuda.is_available():
        raise Exception("need gpu to train!")
    
    args.output_dir= './save/'+group_name+'/'+name_exp+'/'
    assert os.path.exists(args.train_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train_dataset_v1 = torchvision.datasets.ImageFolder(root=args.train_dir,transform=transforms.Compose([transforms.ToTensor(),normalize]))
    train_loader_v1 = torch.utils.data.DataLoader(train_dataset_v1, 
                                               batch_size=args.batch_size, 
                                               shuffle=False, 
                                               num_workers=args.workers, 
                                               pin_memory=True)
    
    ## Initializing teacher models from stage squeeze
    model_teacher = []
    for name in args.teacher_name:
        print("=> loading teacher models '{}'".format(name))
        if name == "ConvNetW128":
            _model = ti_get_network(name, channel=3, num_classes=200, im_size=(64, 64), dist=False)
        else:
            _model = ti_models.model_dict[name](num_classes=200)
        model_teacher.append(_model)
        checkpoint = torch.load(
            os.path.join(args.pre_train_path, "Tiny-ImageNet", name, f"squeeze_{name}.pth"),
            map_location="cpu")
        model_teacher[-1].load_state_dict(checkpoint)
    for _model in model_teacher:
        _model.cuda()
    for _model in model_teacher:
        _model.eval()
    
    full_label = []
    final_images=[]
    hard_labels=[]
    with torch.no_grad():
        for idx, (images,labels) in enumerate(train_loader_v1):
            final_images.append(renormalize_batch(images))
            hard_labels.append(labels)
            print(labels)
            soft_label=[]
            for _model in model_teacher:
                soft_label.append(_model(images.cuda()))
            soft_label = torch.stack(soft_label, 0)
            print(soft_label.shape)
            soft_label = soft_label.mean(0)
            print(soft_label.shape)
            full_label.append(soft_label)
    final_images = torch.cat(final_images, dim=0)
    hard_labels = torch.cat(hard_labels, dim=0)
    full_label = torch.cat(full_label, dim=0)
    print(final_images[0].shape,hard_labels[0],full_label[0].shape)
    print(full_label.shape,final_images.shape,hard_labels.shape)


    


    image_transforms=transforms.Compose([transforms.RandAugment(),
                                         transforms.ToTensor(),
                                         transforms.RandomResizedCrop(size=64,
                                                                      scale=(1/2, 1),
                                                                      interpolation=InterpolationMode.BILINEAR),
                                                                      transforms.RandomHorizontalFlip(),
                                                                      normalize])
    train_dataset = CustomDataset(full_label.cpu(), final_images.cpu(), hard_labels.cpu(),image_transforms)
    print(len(train_dataset))
 
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=args.batch_size, 
                                               shuffle=True, 
                                               num_workers=args.workers, 
                                               pin_memory=True)
    print(len(train_dataset),len(train_loader))

    # for i,(x,y,z) in enumerate(train_loader):
    #     print(x,y.shape,z.shape)
    #     exit()



    # exit()

  
    # Data loading
    # train_dataset = torchvision.datasets.ImageFolder(root=args.train_dir,
    #                                                  transform=transforms.Compose([
    #                                                      transforms.RandAugment(),
    #                                                      transforms.ToTensor(),
    #                                                      transforms.RandomResizedCrop(size=64,
    #                                                                                   scale=(1/2, 1),
    #                                                                                   interpolation=InterpolationMode.BILINEAR),
    #                                                                                   transforms.RandomHorizontalFlip(),
    #                                                                                   normalize]))

    # train_loader = torch.utils.data.DataLoader(train_dataset, 
    #                                            batch_size=args.batch_size, 
    #                                            shuffle=True, 
    #                                            num_workers=args.workers, 
    #                                            pin_memory=True)

    # load validation data
    _ , val_set = load_data(args.val_dir,args.val_dir,args)
    
    val_loader = torch.utils.data.DataLoader(val_set,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=4)
   
    print('val load data successfully')
    
    # load student model
    print("=> loading student model '{}'".format(args.model))
    if "ConvNet" in args.model:
        model = ti_get_network(args.model, channel=3, num_classes=200, im_size=(64, 64), dist=False)
    else:
        model = ti_models.model_dict[args.model](num_classes=200)
    torch.cuda.set_device(gpu)
    model = model.cuda(gpu)
    
    model.train()
    ema_model = EMAMODEL(model)
    args.mode = "fkd_save"
    # model_teacher = []
    # for name in args.teacher_name:
    #     print("=> loading teacher models '{}'".format(name))
    #     if name == "ConvNetW128":
    #         _model = ti_get_network(name, channel=3, num_classes=200, im_size=(64, 64), dist=False)
    #     else:
    #         _model = ti_models.model_dict[name](num_classes=200)
    #     model_teacher.append(_model)
    #     checkpoint = torch.load(
    #         os.path.join(args.pre_train_path, "Tiny-ImageNet", name, f"squeeze_{name}.pth"),
    #         map_location="cpu")
    #     model_teacher[-1].load_state_dict(checkpoint)

    # for _model in model_teacher:
    #     _model.cuda()

    if args.sgd:
        optimizer = torch.optim.SGD(get_parameters(model),
                                    lr=args.sgd_lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(get_parameters(model),
                                      lr=args.adamw_lr,
                                      weight_decay=args.adamw_weight_decay)


    if args.ls_type == "cos":
        scheduler = LambdaLR(optimizer,
                             lambda step: 0.5 * (
                                     1. + math.cos(math.pi * step / (args.st*args.epochs))) if step <= (args.st*args.epochs) else 0,
                             last_epoch=-1)
    elif args.ls_type == "cos2":
        scheduler = LambdaLR(optimizer,
                             lambda step: 0.5 * (
                                     1. + math.cos(math.pi * step / (args.st*args.epochs))) if step <= (args.epochs*5/6) else 0.5 * (
                                     1. + math.cos(math.pi * 5 / (6 * args.st))) * (6*args.epochs-6*step)/(6*args.epochs),
                             last_epoch=-1)
    elif args.ls_type == "alrs":
        scheduler = ALRS(optimizer,decay_rate=args.alrs_dr)
    else:
        scheduler = LambdaLR(optimizer,
                             lambda step: (1.0 - step / (args.st*args.epochs)) if step <= (args.st*args.epochs) else 0, last_epoch=-1)

    args.best_acc1 = 0 # 31.4% -> 34.4% (background)
    args.optimizer = optimizer
    args.scheduler = scheduler
    args.train_loader = train_loader
    args.val_loader = val_loader


    wandb.login(key="085788de93539cf40689d3714b6dc50a54f20a86")
    run=start_subexperiment(args,name_exp,group_name)

    for epoch in range(args.start_epoch, args.epochs):
        print(f"\nEpoch: {epoch}")

        global wandb_metrics
        wandb_metrics = {}

        now_loss = train(model, model_teacher, args, epoch, gpu, ema_model=ema_model)

        wandb.log(wandb_metrics)

        # if epoch % 30 == 0 or epoch == args.epochs - 1:
        ema_model.ema_swap(model)
        top1 = validate(model, args, epoch)
        ema_model.ema_swap(model)
            # top1=validate_ema(ema_model,args,epoch)
            # top1=validate(model,args,epoch)
        # else:
        #     top1 = 0
        wandb_metrics.update({'Validation_acc':top1})
        wandb.log(wandb_metrics)

        if args.ls_type == "alrs":
            scheduler.step(now_loss)
        else:
            scheduler.step()

        # remember best acc@1 and save checkpoint

        is_best = top1 > args.best_acc1 #(True or False)
        args.best_acc1 = max(top1, args.best_acc1) 
        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'best_acc1': args.best_acc1,
                         'optimizer': optimizer.state_dict(),}, is_best, output_dir=args.output_dir)

    run.finish()




def train(model, model_teacher, args, epoch=None, gpu=0, ema_model=None):
   
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    optimizer = args.optimizer
    scheduler = args.scheduler
    loss_function_kl = nn.KLDivLoss(reduction='batchmean')

    model.train()
    t1 = time.time()

    for batch_idx, (images,soft_label, target) in enumerate(args.train_loader):
        images = images.cuda()
        target = target.cuda()
        soft_label = soft_label.cuda()

        images, _, _, _ = mix_aug(images, args)
        optimizer.zero_grad()

       
        output = model(images).float()
        # prec1, prec5 = accuracy(output, target, topk=(1, 5))

        if args.loss_type == "kl":
            output = F.log_softmax(output / args.temperature, dim=1)
            soft_label = F.softmax(soft_label / args.temperature, dim=1)
            loss = loss_function_kl(output, soft_label)
        
        elif args.loss_type == "mse_gt":
            loss = F.mse_loss(output, soft_label) + F.cross_entropy(output, target) * args.ce_weight
        else:
            raise NotImplementedError
        
        loss.backward()
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        n = images.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        optimizer.step()
        if ema_model is not None:
            ema_model.ema_step(decay_rate=args.ema_dr,model=model)
    
    metrics = {"train/loss": objs.avg,
               "train/Top1": top1.avg,
               "train/Top5": top5.avg,
               "train/lr": scheduler.get_last_lr()[0],
               "train/epoch": epoch, }
    wandb_metrics.update(metrics)

    printInfo = 'TRAIN Iter {}: lr = {:.6f},\tloss = {:.6f},\t'.format(epoch, scheduler.get_last_lr()[0], objs.avg) + \
                'Top-1 err = {:.6f},\t'.format(100 - top1.avg) + \
                'Top-5 err = {:.6f},\t'.format(100 - top5.avg) + \
                'train_time = {:.6f}'.format((time.time() - t1))
    print(printInfo)
    t1 = time.time()
    return objs.avg


def validate(model, args, epoch=None):
    
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_function = nn.CrossEntropyLoss()

    model.eval()
    t1 = time.time()
    with torch.no_grad():
        for data, target in args.val_loader:
            target = target.type(torch.LongTensor)
            data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = loss_function(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

    logInfo = 'TEST Iter {}: loss = {:.6f},\t'.format(epoch, objs.avg) + \
              'Top-1 err = {:.6f},\t'.format(100 - top1.avg) + \
              'Top-5 err = {:.6f},\t'.format(100 - top5.avg) + \
              'val_time = {:.6f}'.format(time.time() - t1)
    print(logInfo)

    metrics = {
        'val/loss': objs.avg,
        'val/top1': top1.avg,
        'val/top5': top5.avg,
        'val/epoch': epoch,
    }
    wandb_metrics.update(metrics)

    return top1.avg


def save_checkpoint(state, is_best, output_dir=None, epoch=None):
    if epoch is None:
        path = output_dir + '/' + 'checkpoint.pth.tar'
    else:
        path = output_dir + f'/checkpoint_{epoch}.pth.tar'
    torch.save(state, path)

    if is_best:
        path_best = output_dir + '/' + 'model_best.pth.tar'
        shutil.copyfile(path, path_best)


if __name__ == "__main__":
    main()
