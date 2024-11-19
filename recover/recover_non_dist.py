'''This code is modified from https://github.com/liuzechun/Data-Free-NAS'''
import os
import random
import argparse
import collections
import time
from tqdm import tqdm
import numpy as np
import torchvision.datasets
import time
from PIL import Image
import torch.multiprocessing as mp
import torch
import torch.utils
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.utils.data.distributed
from torchvision import models
import multiprocessing
import torch.distributed as dist
import wandb
from utils import *
import models as ti_models
from baseline import get_network as ti_get_network
from tiny_in_dataset_new import TinyImageNet


normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                std=[0.2302, 0.2265, 0.2262])

def start_subexperiment(args,name,group_name):
    # Start a W&B run for the subexperiment
    run = wandb.init(
        project=args.project_main,  
        name= name,
        reinit=True, 
        group=group_name)  # Optional: Group experiments togethe)
    
    return run

def load_data(traindir, valdir, args):
    normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                     std=[0.2302, 0.2265, 0.2262])
    train_transform = transforms.Compose([transforms.RandomResizedCrop(64),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize])
    dataset = TinyImageNet(traindir, split='train', download=True, transform=train_transform)
    val_transform = transforms.Compose([transforms.ToTensor(),normalize,])
    dataset_test = TinyImageNet(valdir, split='val', download=True, transform=val_transform)
    return dataset, dataset_test

class ApplyTransformToPair:
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, img1, img2):
        seed = random.randint(0,int(1e13))
        torch.manual_seed(seed)
        img1_transformed = self.transform(img1)
        torch.manual_seed(seed)
        img2_transformed = self.transform(img2)
        return img1_transformed, img2_transformed
    
def main_worker(gpu,args, model_teacher, model_verifier, ipc_id_range):
    # args.gpu = gpu
    # gpu=int(args.gpu)
    print(gpu,type(gpu))
    print("Use GPU: {} for training".format(gpu))
    torch.cuda.set_device(gpu)
    ##Teacher Model##
    model_teacher = [_model_teacher.cuda().eval() for _model_teacher in model_teacher]
    for _model_teacher in model_teacher:
        for p in _model_teacher.parameters():
            p.requires_grad = False
    
    model_verifier = model_verifier.cuda()
    model_verifier.eval()
    for p in model_verifier.parameters():
        p.requires_grad = False


    ##epochs and trainig
    save_every = 100
    batch_size = args.batch_size
    best_cost = 1e4
    load_tag_dict = [True for i in range(len(model_teacher))]
    loss_r_feature_layers = [[] for _ in range(len(model_teacher))]
    load_tag = True

    for i, (_model_teacher) in enumerate(model_teacher):
        for name, module in _model_teacher.named_modules():
            full_name = str(args.aux_teacher[i]) + "=" + name
            print(full_name)
          
            if isinstance(module, nn.BatchNorm2d):
                _hook_module = BNFeatureHook(module,save_path=args.statistic_path,
                                            name=full_name,
                                            gpu=gpu,
                                            training_momentum=args.training_momentum,
                                            flatness_weight=args.flatness_weight,
                                            category_aware=args.category_aware)
                
    
                _hook_module.set_hook(pre=True)
                load_tag = load_tag & _hook_module.load_tag
                load_tag_dict[i] = load_tag_dict[i] & _hook_module.load_tag
                loss_r_feature_layers[i].append(_hook_module)

            elif isinstance(module, nn.Conv2d):
                _hook_module = ConvFeatureHook(module, save_path=args.statistic_path,
                                               name=full_name,
                                               gpu=gpu, 
                                               training_momentum=args.training_momentum,
                                               drop_rate=args.drop_rate,
                                               flatness_weight=args.flatness_weight,
                                               category_aware=args.category_aware)
                _hook_module.set_hook(pre=True)
                load_tag = load_tag & _hook_module.load_tag
                load_tag_dict[i] = load_tag_dict[i] & _hook_module.load_tag
                loss_r_feature_layers[i].append(_hook_module)

  
    if not load_tag:
        train_set, val_set = load_data(args.train_data_path,args.train_data_path,args)
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=4)
        
        with torch.no_grad():
            for j, _model_teacher in enumerate(model_teacher):
                if not load_tag_dict[j]:
                    print(f"conduct backbone {args.aux_teacher[j]} statistics")
                    for i, (data, targets) in tqdm(enumerate(train_loader)):
                        data = data.cuda(gpu)
                        targets = targets.cuda(gpu)
                        for _loss_t_feature_layer in loss_r_feature_layers[j]:
                            _loss_t_feature_layer.set_label(targets)
                        _ = _model_teacher(data)
                    
                    for _loss_t_feature_layer in loss_r_feature_layers[j]:
                        _loss_t_feature_layer.save()

        print("Training Statistic Information Is Successfully Saved")
    else:
        print("Training Statistic Information Is Successfully Load")




    initial_img_cache = PreImgPathCache(args.initial_img_dir+'train/',transforms=transforms.Compose([
                                                             transforms.Resize((64,64)),
                                                             transforms.RandomHorizontalFlip(),
                                                             transforms.ToTensor(),
                                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                  std=[0.229, 0.224, 0.225])],))


    for j in range(len(loss_r_feature_layers)):
        for _loss_t_feature_layer in loss_r_feature_layers[j]:
            _loss_t_feature_layer.set_hook(pre=False)

    targets_all_all = torch.LongTensor(np.arange(200))[None, ...].expand(len(ipc_id_range), 200).contiguous().view(-1)
    ipc_id_all = torch.LongTensor(ipc_id_range)[..., None].expand(len(ipc_id_range), 200).contiguous().view(-1)

    total_number = 200 * (ipc_id_range[-1] + 1 - ipc_id_range[0])
    turn_index = torch.LongTensor(np.arange(total_number)).view(len(ipc_id_range), 200) \
        .transpose(1, 0).contiguous().view(-1)
    counter = 0
    ##start by sunny
    ## Nondistributedcode
    new_count=0
    wandb.login(key=args.API_KEY)
    start_index=0
    for zz in range (start_index,total_number,batch_size):
        sub_exp='batch_'+str(int(new_count))
        run= start_subexperiment(args,sub_exp,args.exp_name)
        
        experiment_table = wandb.Table(columns=['Batch_number', "Total_loss","loss_ema_ce",'r_feature_loss','ce_loss','VerifierAccuracy_last','ipc'])
        sub_turn_index= turn_index[zz:zz+batch_size]
        targets = targets_all_all[sub_turn_index].cuda(gpu)
        ipc_ids =ipc_id_all[sub_turn_index].cuda(gpu)
        data_type=torch.float
        print(f"In GPU {gpu}, targets is set as: \n{targets}\n, ipc_ids is set as: \n{ipc_ids}")
        
        inputs = torch.stack([initial_img_cache.random_img_sample(_target) for _target in targets.tolist()],0).to(f'cuda:{gpu}').to(data_type)
        inputs.requires_grad_(True)

        iterations_per_layer = args.iteration
        optimizer = optim.Adam([inputs], lr=args.lr, betas=[0.5, 0.9], eps=1e-8)
        lr_scheduler = lr_cosine_policy(args.lr, 0, iterations_per_layer)  # 0 - do not use warmup
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()
        inputs_ema = EMA(alpha=args.ema_alpha, initial_value=inputs)
    
        for iteration in range(iterations_per_layer):

            lr_scheduler(optimizer, iteration, iteration)

            aug_function = ApplyTransformToPair(transforms.Compose([
                transforms.RandomResizedCrop(64,scale=(0.5, 1)),
                transforms.RandomHorizontalFlip()]))
            
            inputs_jit, inputs_ema_jit = aug_function(inputs, inputs_ema.value)

            id = counter % len(model_teacher)

            for mod in loss_r_feature_layers[id]:
                mod.set_label(targets)
            
            counter += 1
            optimizer.zero_grad()
            
            for (idx, mod) in enumerate(loss_r_feature_layers[id]):
                mod.set_ori()

            if torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs!")
            model_out = torch.nn.DataParallel( model_teacher[id])
            sub_outputs=model_out(inputs_jit)
            # sub_outputs = model_teacher[id](inputs_jit)
            
            # R_cross classification loss
            loss_ce = criterion(sub_outputs, targets)

            # R_feature loss
            
            rescale=[]
            for name in loss_r_feature_layers[id]:
                if isinstance(name.module,nn.Conv2d):
                    rescale.append(args.conv_weight)
                elif isinstance(name.module,nn.BatchNorm2d):
                    rescale.append(args.batch_weight)
            if args.first_multiplier == True:
                rescale[0]=10
        
            loss_r_feature = sum(
                [mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers[id])])
            ##Flatness_loss
            if args.flatness:
                with torch.no_grad():
                    for (idx, mod) in enumerate(loss_r_feature_layers[id]):
                        mod.set_return()
                    ema_sub_outputs = model_teacher[id](inputs_ema_jit)
                    for (idx, mod) in enumerate(loss_r_feature_layers[id]):
                        mod.remove_return()
                    inputs_ema.ema_update(inputs)
                loss_ema_ce = F.kl_div(torch.log_softmax(sub_outputs / 4, dim=1),
                                        torch.softmax(ema_sub_outputs / 4, dim=1))
            else:
                loss_ema_ce = torch.Tensor([0.]).to(inputs_jit.device)
        
            loss_aux = args.r_loss * loss_r_feature + loss_ema_ce * args.flatness_weight
            loss = loss_ce + loss_aux

            if iteration % save_every == 0 :
                print("------------iteration {}----------".format(iteration))
                print("------------sub_exp {}----------".format(sub_exp))
                print("total loss", loss.item())
                print("loss_r_feature", loss_r_feature.item())
                print("loss_ema_ce", loss_ema_ce.item())
                print("main criterion",loss_ce.item())
            
            wandb.log({"iteration": iteration,"total_loss":loss.item(),"loss_r_feature":loss_r_feature.item(),"loss_ema_ce":loss_ema_ce.item(),
                        'loss_ce_main': loss_ce.item(),'id':id})

        # do image update
            loss.backward()
            optimizer.step()

            # clip color outlayers
            inputs.data = clip(inputs.data)
            if best_cost > loss.item():
                best_inputs = inputs.data.clone()
                best_cost = loss.item()


        if args.store_best_images:
            best_inputs = inputs.data.clone()  # using multicrop, save the last one
            best_inputs = denormalize(best_inputs)
            inputs = normalize((best_inputs * 255).int() / 255)
            print("Testing...")
            acc_top_1_veri= validate(inputs,targets,model_verifier)
            save_images(args, best_inputs, targets, ipc_ids)
        
        if iterations_per_layer > 0:
            experiment_table.add_data(sub_exp,loss.item(),loss_ema_ce.item(),loss_r_feature.item(),loss_ce.item(),acc_top_1_veri,args.ipc_number)
            wandb.log({"experiment_results": experiment_table})
        new_count+=1
        # to reduce memory consumption by states of the optimizer we deallocate memory
        optimizer.state = collections.defaultdict(dict)
        torch.cuda.empty_cache()
    run.finish()



def save_images(args, images, targets, ipc_ids):
    ipc_id_range = ipc_ids
    for id in range(images.shape[0]):
        if targets.ndimension() == 1:
            class_id = targets[id].item()
        else:
            class_id = targets[id].argmax().item()

        if not os.path.exists(args.syn_data_path):
            os.mkdir(args.syn_data_path)

        # save into separate folders
        dir_path = '{}/new{:03d}'.format(args.syn_data_path, class_id)
        place_to_store = dir_path + '/class{:03d}_id{:03d}.png'.format(class_id, ipc_id_range[id])
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(place_to_store)



def validate(input, target, model):
    model.eval()
    with torch.no_grad():
        output = model(input.float().cuda())
        target= torch.tensor(target).cuda()
        total_acc = (target == output.argmax(1)).float().sum().item()
        total_number = target.shape[0]
        top_1_acc_val = round(total_acc * 100 / total_number, 3)
    print(f"Top-1 Validation Accuracy: {top_1_acc_val}%")
    return top_1_acc_val

    


def main_syn():

    parser = argparse.ArgumentParser(
        "G-VBSM: applying generalized matching for data condensation")
    """Data save flags"""
    parser.add_argument('--flatness', action='store_true',
                        help='encourage the flatness or not')
    parser.add_argument('--flatness-weight', type=float, default=1.,
                        help='the weight of flatness weight')
    parser.add_argument('--ema_alpha', type=float, default=0.99,
                        help='the weight of EMA learning rate')
    parser.add_argument('--conv_weight', type=int, default=1,
                        help='the weight of EMA learning rate')
    parser.add_argument('--batch_weight', type=int, default=1,
                        help='the weight of EMA learning rate')
    parser.add_argument('--aux-teacher', nargs='+', 
                        default=["ResNet18", "ConvNetW128", "MobileNetV2", "WRN_16_2", "ShuffleNetV2_0_5"],
                        help='define the teacher models')
    parser.add_argument('--exp-name', type=str, default='test',
                        help='name of the experiment, subfolder under syn_data_path')
    parser.add_argument('--ipc-number', type=int, default=10, help='the number of each ipc')
    parser.add_argument('--syn-data-path', type=str,
                        default='./syn_data', help='where to store synthetic data')
    parser.add_argument('--syn-data-path-best', type=str,
                        default='./syn_data_best', help='where to store synthetic data')
    parser.add_argument('--pre-train-path', type=str,
                        default='../squeeze/squeeze_wo_ema/',
                        help='where to load the pre-trained backbone')
    parser.add_argument('--store-best-images', action='store_true',
                        help='whether to store best images')
    """Optimization related flags"""
    parser.add_argument('--batch-size', type=int,
                        default=100, help='number of images to optimize at the same time')
    parser.add_argument('--gpu-id', type=str, default='0,1,2')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--initial-img-dir', type=str, default="./exp/../", 
                       help="imgs used for initialization")
    parser.add_argument('--iteration', type=int, default=1000,
                        help='num of iterations to optimize the synthetic data')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate for optimization')
    parser.add_argument('--training-momentum', type=float, default=0.8,
                        help="$\alpha$ in our paper, controls the form of score distillation sampling")
    parser.add_argument('--drop-rate', type=float, default=0.4,
                        help="$\beta_\textrm{dr}$ in our paper, controls the efficiency of GSM")
    parser.add_argument('--jitter', default=32, type=int, help='random shift on the synthetic data')
    parser.add_argument('--category-aware', default="global", type=str, help='category-aware matching (local or global)')
    parser.add_argument('--r-loss', type=float, default=0.05,
                        help='coefficient for BN and Conv feature distribution regularization')
    parser.add_argument('--first-multiplier', action='store_true',
                        help='encourage the flatness or not')
    parser.add_argument('--nuc-norm', type=float, default=0.00001,
                        help='coefficient for total variation Nuclear loss')
    """Model related flags"""
    parser.add_argument('--arch-name', type=str, default='resnet18',
                        help='arch name from pretrained torchvision models')
    parser.add_argument('--tau', type=float, default=4.0, help='the temperature of nuc norm')
    parser.add_argument('--verifier', action='store_true',
                        help='whether to evaluate synthetic data with another model')
    parser.add_argument('--verifier-arch', type=str, default='mobilenet_v2',
                        help="archx name from torchvision models to act as a verifier")
    parser.add_argument('--train-data-path', type=str, default='./tiny_imagenet/train',
                        help="the path of the Tiny-ImageNet's training set")
    parser.add_argument('--statistic-path', type=str, default='./statistic',
                        help="the path of the statistic file")
    
    args = parser.parse_args()
    args.project_main='EDC_recover_paper'
    
    val='['
    for name in args.aux_teacher:
        val+=name[0:3]
    val+=']'

    args.exp_name= 'd_'+'tinyim'+'_it_'+str(args.iteration)+'_ipc_'+str(args.ipc_number)+"_flat_"+str(args.flatness)+'_bs_'+str(args.batch_size)+ '_lr_'+ str(args.lr)+'_ca_'+str(args.category_aware)+'_tea_'+str(val)+'_conw_'+str(args.conv_weight)+'_batw_'+str(args.batch_weight)+'_mom_'+str(args.training_momentum)+'_fmul_'+str(args.first_multiplier)
    args.syn_data_path = os.path.join(args.syn_data_path, args.exp_name)
    args.syn_data_path_best = os.path.join(args.syn_data_path_best, args.exp_name)
    if not os.path.exists(args.syn_data_path):
        os.makedirs(args.syn_data_path)


    model_teacher = []
    for name in args.aux_teacher:
        if name == "ConvNetW128":
            model = ti_get_network(name, channel=3, num_classes=200, im_size=(64, 64), dist=False)
        elif name == "efficientnet_b0":
            model = models.efficientnet_b0(weights=None)
            num_classes = 200  
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        else:
            model = ti_models.model_dict[name](num_classes=200)
        model_teacher.append(model)
        checkpoint = torch.load(
            os.path.join(args.pre_train_path, "Tiny-ImageNet", name, f"squeeze_{name}.pth"),
            map_location="cpu")
        model_teacher[-1].load_state_dict(checkpoint)

    model_verifier = ti_models.model_dict['ResNet18'](num_classes=200)
    model_veri_checkpoint= torch.load(
            os.path.join(args.pre_train_path, "Tiny-ImageNet", "ResNet18", f"squeeze_ResNet18.pth"),
            map_location="cpu")
    model_verifier.load_state_dict(model_veri_checkpoint)

    ipc_id_range = list(range(0, args.ipc_number))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    print(args.gpu_id,type(args.gpu_id))
    args.API_KEY = os.environ.get("API_KEY")
    main_worker(args.gpu,args, model_teacher, model_verifier, ipc_id_range)

if __name__ == '__main__':
    main_syn()
