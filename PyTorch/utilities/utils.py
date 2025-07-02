import numpy as np
import torch
import os
import glob
from shutil import rmtree
import time
from pathlib import Path
import logging
from datetime import datetime
import argparse
import sys
sys.path.append("..")
import wandb
from arch import simple_multi_head_jd_model
import yaml
from types import SimpleNamespace

def export_image(filename, img):
    np.save(filename, img)


def build_pattern(order = "RGGB", filter_size = 1, rows = 300, cols = 300, pattern_dict = {"R":0, "G": 1, "B": 2}):
    pattern_tile = torch.empty((filter_size*2, filter_size*2), dtype=torch.int64)
    for row in range(2):
        for col in range(2):
            pattern_index = row*2 + col
            channel_letter = order[pattern_index]
            pattern_channel = pattern_dict[channel_letter]
            pattern_tile[row*filter_size:(row+1)*filter_size, col*filter_size:(col+1)*filter_size] = pattern_channel
    pattern = torch.tile(pattern_tile, (rows//(filter_size*2), cols//(filter_size*2)))
    return pattern


def construct_mosaic(image, order = "RGGB", filter_size = 1):
    pattern = build_pattern(order=order, filter_size=filter_size, rows=image.shape[0], cols=image.shape[1])
    out = torch.gather(input=image, dim=2, index=pattern.unsqueeze_(dim=-1)).squeeze()
    return pattern, out




def calc_model_size(model):
    # From: https://discuss.pytorch.org/t/finding-model-size/130275
    # ~~ in @ptrblck we trust ~~
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / (1024 ** 2) # for MB

    return size_all_mb

#move this to a utils file
def get_freer_gpu(num_gpus):
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_available = np.array([int(x.split()[2]) for x in open('tmp', 'r').readlines()])

        #this means I am on the other server
        if len(memory_available) == 0:
            os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
            memory_used = np.array([int(x.split()[2]) for x in open('tmp', 'r').readlines()])
            return np.argsort(memory_used)[0:num_gpus]
        else:
            return np.argsort(-1*memory_available)[0:num_gpus]
    
#from deep_architect with MIT License
def get_total_num_gpus():
    try:
        import subprocess
        n = len(subprocess.check_output(['nvidia-smi','-L']).decode('utf-8').strip().split('\n'))
    except OSError:
        n = 0
    return n 

#make this a function
def delete_old_wandb_files(owner_name):
    for file_location in glob.glob(f'/tmp/*wandb*'):
        # file_time is the time when the file is modified
        file_time = os.stat(file_location).st_mtime

        # if a file is modified before N days then delete it
        N=7
        current_time = time.time()
        day = 86400 #seconds in a day

        path = Path(file_location)
        owner = path.owner()
        if (owner == owner_name):
            if(file_time < (current_time - day*N)):
                print(f" Delete : {file_location}")
                rmtree(file_location)


def get_device(args):
    gpus_chosen = get_freer_gpu(args.num_gpus).tolist()
    logging.info(f'Using devices {gpus_chosen}')

    all_gpus = [i for i in range(get_total_num_gpus())]
    all_gpus.sort()

    logging.info(f'ALL GPUS {all_gpus}')

    #gpus_chosen = [0]

    os.environ["CUDA_VISIBLE_DEVICES"]=",".join(map(str, all_gpus))
    os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] ="50"
    device = f"cuda:{gpus_chosen[0]}"
    return gpus_chosen, device

def get_args():
    parser = argparse.ArgumentParser(description="YAML config loader")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    return SimpleNamespace(**config)

def setup_wandb_config(args, net, macs, params):

    model_size = calc_model_size(net)
    
    wandb.config = {
        "learning_rate": args.lr,
        "patch_size": args.patchsz,
        "workers": args.workers,
        "model": args.model,
        "head": args.head,
        "single_head": args.single_head,
        "nona_head": args.nona_head,
        "support_width": args.support_width,
        "backbone": args.backbone,
        "size_mb": model_size,
        "batch_size": args.batchsize,
        "normalize_filter": args.normalize_filter,
        "pattern": args.pattern,
        "remosaic": args.remosaic,
        "loaded_model_path": args.load,
        "iso": args.iso,
        "job_id": args.job_id,
        "backbone_iso": args.backbone_iso,
        "depth": args.depth,
        "patch_dir": args.patchdir,
        "hard_patches": args.hard_patches,
        "plot_images": args.plot_images, 
        "noisy_viz": args.noisy_viz,
        "dropout": args.dropout,
        "mask_interpolation": args.mask_interpolation,
        "gauss_sigma": args.gauss_sigma,
        "range_sigma": args.range_sigma,
        "weight_scale": args.weight_scale,
        "denoise": args.denoise,
        "macs": macs,
        "params": params,
        "alpha": args.alpha,
        }

    currentSecond= datetime.now().second
    currentMinute = datetime.now().minute
    currentHour = datetime.now().hour
    currentDay = datetime.now().day
    currentMonth = datetime.now().month
    currentYear = datetime.now().year

    
    
    run_name = f"{args.model}_{args.pattern}_{args.backbone}_rm-{args.remosaic}_iso{args.iso}_bi_{args.backbone_iso}_d{args.dropout}_{currentMonth}-{currentDay}-{currentHour}-{currentMinute}-{currentSecond}"
    return run_name

def get_model(args, device):
    discriminator = None
    teacher = None
    if args.model == "simp_multi":
        net = simple_multi_head_jd_model.SIMP_MULTI_JD()

    return net, discriminator, teacher

def output_rescaling_quantization(imgs):
    saved_device = imgs.device
    imgs = np.clip(((imgs*0.5+0.5)*13496).cpu().numpy(), 0, 13496).astype(np.uint16)
    imgs = torch.from_numpy(imgs.astype(np.int32)).to(saved_device)
    return imgs

def generate_output_images(out_imgs_dir, imgs, psnrs, filenames):

    out = []
    for i, image in enumerate(imgs[:].cpu().detach().numpy()):
        grandparent_directory, parent_directory, filename = Path(filenames[i]).parts[-3:]
        
        image = image/13496 #renormalize
        image = np.clip(image, 0, 1)
        patch_gamma = np.moveaxis(image, [0,1,2], [2,0,1]) #no gamma
        if image.shape[1] == 44:
            parts = filename.split('_')
            psnr_round = format(psnrs[i], '.2f')
            parts[-2] = f'p{psnr_round}'
            filename = '_'.join(parts)
            filename = filename[:-4]+".npy"
            #upscale_factor = 4
            #patch_upscaled = patch_gamma.repeat(upscale_factor, 1).repeat(upscale_factor, 0)
        else:
            parts = filename.split('_')
            psnr_round = format(psnrs[i], '.2f')
            parts.insert(2, f'p{psnr_round}')
            filename = '_'.join(parts)
            filename = filename[:-4]+".npy"
        
        full_path = os.path.join(out_imgs_dir, grandparent_directory, parent_directory, filename)

        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        np.save(full_path, patch_gamma[:,:,::-1])  #flip channel order
    return out

def set_job_id(args):
    if args.job_id is None:
        args.job_id = int(time.time())
    return args

def generate_wandb_images(imgs, filenames, psnrs):
    out = []
    for i, image in enumerate(imgs[:2].cpu().detach().numpy()): #only do two patches
        image = image/13496 #renormalize
        image = np.clip(image, 0, 1)
        patch_gamma = np.moveaxis(image, [0,1,2], [2,0,1]) **(1/2.2)
        upscale_factor = 2
        patch_upscaled = patch_gamma.repeat(upscale_factor, 1).repeat(upscale_factor, 0)
    return out




def save_checkpoint(checkpoint_name, dir_checkpoint, net, optimizer, scheduler, discriminator, optimizer_discriminator, scheduler_discriminator):
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint, exist_ok=True)
        logging.info('Created checkpoint directory')
    state_dict = {
    'net_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    }

    if discriminator is not None:
        state_dict["discriminator_state_dict"] = discriminator.state_dict()
        state_dict["optimizer_discriminator_state_dict"] = optimizer_discriminator.state_dict()
        state_dict["scheduler_discriminator_state_dict"] = scheduler.state_dict()

    torch.save(state_dict, dir_checkpoint + f'{checkpoint_name}')
    
