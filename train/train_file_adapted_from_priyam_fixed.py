import os
import sys
import math
import time
import shutil
import argparse
import numpy as np
import wandb
import time
from timm import create_model
from copy import deepcopy
import torch.distributed as dist
sys.path.append('../')
from torchvision import models
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
from torch.optim.swa_utils import AveragedModel
from relabel.utils_fkd import mix_aug
from torch.utils.data import Dataset, DataLoader
torch.cuda.empty_cache()


class CustomDataset(Dataset):
    def __init__(self,final_images, hard_labels,full_label,transform):
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

def start_subexperiment(args,name,group_name):
    # Start a W&B run for the subexperiment
    run = wandb.init(
        project=args.wandb_project,  
        name= name,
        reinit=True, 
        group=group_name)  
    
    return run

class ExponentialMovingAverage(AveragedModel):
    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)

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
    parser.add_argument('--output-dir', type=str, default='./save_fixed/', help='path to output dir')
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
    parser.add_argument('--adamw-weight-decay', type=float,default=0.0005, help='adamw weight decay')
    parser.add_argument('--ce-weight', type=float,default=0.1, help='the weight og cross-entropy loss')
    parser.add_argument('--gpu-id', type=str,default="0,1,2", help='the id of gpu used')
    parser.add_argument('--model', type=str,default='ResNet18', help='student model name')
    parser.add_argument('--shuffle-patch', default=False, action='store_true',help='if use shuffle-patch')
    parser.add_argument('--keep-topk', type=int, default=200,help='keep topk logits for kd loss')
    parser.add_argument('-T', '--temperature', type=float,default=3.0, help='temperature for distillation loss')
    parser.add_argument('--fkd-path', type=str,default=None, help='path to fkd label')
    parser.add_argument('--wandb-project', type=str,default='EDC_TIN_fixed_transferability', help='wandb project name')
    parser.add_argument('--wandb-api-key', type=str,default=None, help='wandb api key')
    parser.add_argument('--mix-type', default=None, type=str,choices=['mixup', 'cutmix', None], help='mixup or cutmix or None')
    parser.add_argument('--fkd_seed', default=42, type=int,help='seed for batch loading sampler')
    parser.add_argument('--world-size', default=1, type=int,help='number of nodes for distributed training')
    parser.add_argument('--teacher-name', nargs='+', default=["ResNet18", "ConvNetW128", "MobileNetV2", "WRN_16_2"], help='define the teacher models')
    parser.add_argument('--soft-augmentation', action='store_true', help="Enable the flag")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument("--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)")
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    args = parser.parse_args()
    return args

class ExponentialMovingAverage(AveragedModel):
    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)

def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    main_worker(args.gpu,args)

def fixed_soft_labels(train_loader_v1,model_teacher):
    full_label = []
    final_images=[]
    hard_labels=[]
    with torch.no_grad():
        for idx, (images,labels) in enumerate(train_loader_v1):
            final_images.append(renormalize_batch(images))
            hard_labels.append(labels)
            soft_label=[]
            for _model in model_teacher:
                soft_label.append(_model(images.cuda()))
            soft_label = torch.stack(soft_label, 0)
            soft_label = soft_label.mean(0)
            full_label.append(soft_label)
    final_images = torch.cat(final_images, dim=0)
    hard_labels = torch.cat(hard_labels, dim=0)
    full_label = torch.cat(full_label, dim=0)
    return final_images, hard_labels, full_label


def main_worker(gpu,args):

    torch.cuda.set_device(gpu) 
    val='['
    for name in args.teacher_name:
        val+=name[0:3]
    val+=']'

    group_name= args.train_dir.split('/')[-1]
    name_exp= 'Relabel'+'_teacher_'+str(val)+'_epc_'+str(args.epochs)+'_stumod_'+args.model+'_augment_'+str(args.soft_augmentation)
    if not torch.cuda.is_available():
        raise Exception("need gpu to train!")
    
    args.output_dir= './save_fixed/'+group_name+'/'+name_exp+'/'
    assert os.path.exists(args.train_dir)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train_dataset_v1 = torchvision.datasets.ImageFolder(root=args.train_dir,transform=transforms.Compose([transforms.ToTensor(),normalize]))
    train_loader_v1 = torch.utils.data.DataLoader(train_dataset_v1, 
                                               batch_size=args.batch_size, 
                                               shuffle=False, 
                                               num_workers=args.workers, 
                                               pin_memory=True)

  
    # Data loading
    if args.soft_augmentation is True:
         transform=transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomResizedCrop(size=64,
                                                                 scale=(1/2, 1),
                                                                 interpolation=InterpolationMode.BILINEAR),
                                                                 transforms.RandomHorizontalFlip(),
                                                                                      normalize])
    else:
        transform=transforms.Compose([transforms.RandAugment(),
                                      transforms.ToTensor(),
                                      transforms.RandomResizedCrop(size=64,
                                                                   scale=(1/2, 1),
                                                                   interpolation=InterpolationMode.BILINEAR),
                                                                   transforms.RandomHorizontalFlip(),
                                                                   normalize])


    # train_dataset = torchvision.datasets.ImageFolder(root=args.train_dir,
    #                                                  transform=transform)
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

    # image_transforms=transforms.Compose([transforms.RandAugment(),
    #                                      transforms.ToTensor(),
    #                                      transforms.RandomResizedCrop(size=64,
    #                                                                   scale=(1/2, 1),
    #                                                                   interpolation=InterpolationMode.BILINEAR),
    #                                                                   transforms.RandomHorizontalFlip(),
    #                                                                   normalize])
    
    # final_images, hard_labels, full_label = fixed_soft_labels(train_loader_v1,model_teacher)
    # train_dataset = CustomDataset(full_label.cpu(), final_images.cpu(),hard_labels.cpu(),transform)
    # train_loader = torch.utils.data.DataLoader(train_dataset, 
    #                                            batch_size=args.batch_size, 
    #                                            shuffle=True, 
    #                                            num_workers=args.workers, 
    #                                            pin_memory=True)
    
    # load student model
    print("=> loading student model '{}'".format(args.model))
    if "ConvNet" in args.model:
        model = ti_get_network(args.model, channel=3, num_classes=200, im_size=(64, 64), dist=False)
    elif "DeiT-Tiny" in args.model:
        model = create_model('deit_tiny_patch16_224', pretrained= False)

        # Update the model's positional embeddings
        pos_embed = model.pos_embed  # Positional embeddings
        cls_token = pos_embed[:, 0, :]  # CLS token
        pos_tokens = pos_embed[:, 1:, :]  # Patch embeddings

        # Adjust positional embeddings for 4x4 grid (64x64 -> 16 patches)
        pos_tokens = pos_tokens.reshape(1, 14, 14, -1)  # Original grid (14x14 for 224x224 images)
        pos_tokens = nn.functional.interpolate(
        pos_tokens.permute(0, 3, 1, 2),  # [1, 192, 14, 14]
        size=(4, 4),  # New grid size (4x4 for 64x64 images)
        mode='bicubic',
        align_corners=False
        ).permute(0, 2, 3, 1).reshape(1, 16, -1)  # [1, 16, 192]

        # Concatenate CLS token and new positional encodings
        new_pos_embed = torch.cat([cls_token.unsqueeze(1), pos_tokens], dim=1)  # [1, 17, 192]
        model.pos_embed = nn.Parameter(new_pos_embed)
        model.patch_embed.img_size = (64, 64)  # Update the expected image size
        model.patch_embed.grid_size = (4, 4)
        num_classes = 200
        model.head = nn.Linear(model.head.in_features, num_classes)
        
    else:
        model = ti_models.model_dict[args.model](num_classes=200)
    
    torch.cuda.set_device(gpu)
    model = model.cuda(gpu)
    model.train()
    ema_model = ExponentialMovingAverage(model,decay=0.99,device='cuda')
    args.mode ='fkd_save'
    model_teacher = []

    for name in args.teacher_name:
        print("=> loading teacher models '{}'".format(name))
        if name == "ConvNetW128":
            _model = ti_get_network(name, channel=3, num_classes=200, im_size=(64, 64), dist=False)
        elif name == "efficientnet_b0":
            _model = models.efficientnet_b0(weights=None)
            num_classes = 200  
            _model.classifier[1] = torch.nn.Linear(_model.classifier[1].in_features, num_classes)
        else:
            _model = ti_models.model_dict[name](num_classes=200)
        model_teacher.append(_model)
        checkpoint = torch.load(os.path.join(args.pre_train_path, "Tiny-ImageNet", name, f"squeeze_{name}.pth"),map_location="cpu")
        model_teacher[-1].load_state_dict(checkpoint)
    for _model in model_teacher:
        _model.cuda()
    for _model in model_teacher:
        _model.eval()


    final_images, hard_labels, full_label = fixed_soft_labels(train_loader_v1,model_teacher)
    train_dataset = CustomDataset(final_images.cpu(),hard_labels.cpu(),full_label.cpu(),transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=args.batch_size, 
                                               shuffle=True, 
                                               num_workers=args.workers, 
                                               pin_memory=True)
    print(final_images.shape,hard_labels.shape,full_label.shape)

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
            optimizer, schedulers=[warmup_lr_scheduler, scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = scheduler


    args.best_acc1 = 0  
    args.optimizer = optimizer
    args.scheduler = lr_scheduler
    args.train_loader = train_loader
    args.val_loader = val_loader

    wandb.login(key=args.wandb_api_key)
    run=start_subexperiment(args,name_exp,group_name)

    single_epoch_time=0
    experiment_table = wandb.Table(columns=['experiment_name','group_name','teacher_models','Student_model','ipc','epochs','time_taken','max_val_accuracy','softaugmentation'])
    
    for epoch in range(args.start_epoch, args.epochs):
        
        print(f"\nEpoch: {epoch}")

        global wandb_metrics
        wandb_metrics = {}
        now_loss,time_done = train(model, model_teacher, args, epoch, gpu, ema_model=ema_model)
        top1 = validate(ema_model, args, epoch)
        wandb_metrics.update({'Validation_acc':top1})
        wandb.log(wandb_metrics)
        if args.ls_type == "alrs":
            lr_scheduler.step(now_loss)
        else:
            lr_scheduler.step()
        
        single_epoch_time+=time_done
        # remember best acc@1 and save checkpoint
        is_best = top1 > args.best_acc1 #(True or False)
        args.best_acc1 = max(top1, args.best_acc1) 
        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'best_acc1': args.best_acc1,
                         'optimizer': optimizer.state_dict(),}, is_best, output_dir=args.output_dir)
    
    
    experiment_table.add_data(name_exp,group_name,str(args.teacher_name),args.model,args.ipc,args.epochs,single_epoch_time,args.best_acc1,args.soft_augmentation)
    wandb.log({"experiment_results": experiment_table})
    run.finish()




def train(model, model_teacher, args, epoch=None, gpu=0, ema_model=None):
   
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    optimizer = args.optimizer
    scheduler = args.scheduler
    loss_function_kl = nn.KLDivLoss(reduction='batchmean')

    model.train()

    for _model in model_teacher:
        _model.eval()
    
    t1 = time.time()
    
    for batch_idx, (images,soft_label,target) in enumerate(args.train_loader):
        images = images.cuda()
        target = target.cuda()
        soft_label=soft_label.cuda()

        if args.soft_augmentation is False:
            images, _, _, _ = mix_aug(images, args)
        
        optimizer.zero_grad()

        output = model(images).float()
      
        if args.loss_type == "kl":
            output = F.log_softmax(output / args.temperature, dim=1)
            soft_label = F.softmax(soft_label / args.temperature, dim=1)
            loss = loss_function_kl(output, soft_label)
        elif args.loss_type == "mse_gt":
            loss = F.mse_loss(output, soft_label) + F.cross_entropy(output, target) * args.ce_weight
        else:
            raise NotImplementedError
        loss.backward()
        optimizer.step()
        
        if ema_model:
            ema_model.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                ema_model.n_averaged.fill_(0)
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        n = images.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

    
    metrics = {"train/loss": objs.avg,
               "train/Top1": top1.avg,
               "train/Top5": top5.avg,
               "train/lr": scheduler.get_last_lr()[0],
               "train/epoch": epoch}
    
    wandb_metrics.update(metrics)
    printInfo = 'TRAIN Iter {}: lr = {:.6f},\tloss = {:.6f},\t'.format(epoch, scheduler.get_last_lr()[0], objs.avg) + \
                'Top-1 err = {:.6f},\t'.format(100 - top1.avg) + \
                'Top-5 err = {:.6f},\t'.format(100 - top5.avg) + \
                'train_time = {:.6f}'.format((time.time()-t1 ))
    print(printInfo)
    return objs.avg , time.time()-t1 


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
        'val/epoch': epoch}
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
