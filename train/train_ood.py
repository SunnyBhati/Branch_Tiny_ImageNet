import os
import sys
import warnings
import torch
import torch.utils.data
import torchvision
sys.path.append('../')
import utils_ood as utils
from torch import nn
import relabel.models as ti_models
import torchvision.transforms as transforms
from tiny_in_dataset_new import TinyImageNet
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target.reshape(-1))

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    # print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg


def main(args):
    global best_acc1
    best_acc1 = 0

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
    
    val_transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], 
                             std=[0.2302, 0.2265, 0.2262])])

    print("Creating model")
    def create_model(model_name, path=None):
        model = ti_models.model_dict[model_name](num_classes=200)
        # model = torchvision.models.get_model(model_name, weights=None, num_classes=200)
        # model.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        # model.maxpool = nn.Identity()
        if path is not None:
            checkpoint = torch.load(path, map_location="cpu")
            if "model" in checkpoint:
                checkpoint = checkpoint["model"]
            elif "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            if "module." in list(checkpoint.keys())[0]:
                checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            model.load_state_dict(checkpoint)
        model.to(device)
        return model

    criterion = nn.CrossEntropyLoss()
    
    if args.ckpt_wandb_id is not None:
        ckpt_path = None
        for path in os.listdir(args.ckpt_folder):
            if path.startswith(args.ckpt_wandb_id):
                ckpt_path = os.path.join(args.ckpt_folder, path, 'checkpoint_best.pth.tar')
        if ckpt_path is None:
            raise
    else:
        ckpt_path = args.ckpt_folder

    model = create_model(args.model, path=ckpt_path)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    avg_method_accs = []
    for method in ['brightness',  'contrast',  'defocus_blur',  'elastic_transform',  'fog',  'frost',  'gaussian_noise',  'glass_blur',  'impulse_noise',  'jpeg_compression',  'motion_blur',  'pixelate',  'shot_noise',  'snow',  'zoom_blur']:
        avg_severity_accs = []
        for severity in range(1, 6):
            dset = ImageFolder(root=os.path.join('/data/sunny/EDC/Branch_Tiny_ImageNet/dataset/Tiny-ImageNet-C/', method, str(severity)), transform=val_transform)
            loader = DataLoader(dset, batch_size=256, shuffle=False, num_workers=8)
            acc = evaluate(model, criterion, loader, device=device)
            avg_severity_accs.append(acc)
        # print(f"{args.ckpt_wandb_id} TinyImageNet-{method} Avg Severity Acc: {sum(avg_severity_accs)/len(avg_severity_accs):.2f}, Individual Accs: {avg_severity_accs}")
        avg_method_accs += avg_severity_accs
    print(f"{args.ckpt_wandb_id} TinyImageNet Avg Method Acc: {sum(avg_method_accs)/len(avg_method_accs):.2f}")
            


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Classification Training on Tiny ImageNet", add_help=add_help)
    parser.add_argument("--model", default="ResNet18", type=str, help="model name")
    parser.add_argument("--ckpt_wandb_id", default=None, type=str)
    parser.add_argument("--ckpt_folder", default=None, type=str)
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)")
    parser.add_argument("--sync-bn",dest="sync_bn",help="Use sync batch norm",action="store_true")
    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only.")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    args.ckpt_folder= '/data/sunny/EDC/Branch_Tiny_ImageNet/train/save/d_tinyim_it_2000_ipc_10_flat_True_bs_100_lr_0.05_ca_global_tea_[ResMobShueff]_conw_1_batw_1_mom_0.99_fmul_True/Relabel_teacher_[ResMobShueff]_epc_300_stumod_ResNet18_augment_False/model_best.pth.tar'
    main(args)