"""
Copyright (c) 2025 Samsung Electronics Co., Ltd.

Author(s):
SaiKiran Tedla
Abhijith Punnappurath
Luxi Zhao
Michael S. Brown

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

"""


import logging
import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import pickle
import torch
from torch import optim
from tqdm import tqdm
import torch.nn as nn
from utilities.utils import get_device, get_model, delete_old_wandb_files, get_args, output_rescaling_quantization, set_job_id, setup_wandb_config, generate_output_images, save_checkpoint
from utilities.loss_func import mae_loss, mse_loss,psnr_per_image, mse_validation_loss, mse_loss_sum_across_batch, ssim_per_image, deltae_per_image
from torch.utils.data import DataLoader
from utilities.dataset import BasicDataset

import wandb
def train_net(net,
              model_name,
              discriminator,
              pattern,
              remosaic,
              iso,
              device,
              epochs=10,
              batch_size=32,
              lr=0.0001,
              lrdf=0.5,
              lrdp=25,
              chkpointperiod=1,
              trimages=0,
              patchsz=144,
              patchnum=4,
              validationFrequency=4,
              dir_img='../Scenes_npy_split/',
              num_workers = 4,
              save_cp=True,
              run_name = "",
              dir_checkpoint = None,
              job_id=None,
              hard_patches=False,
              dropout = 0,
              odir = None,):

    if dir_checkpoint is None:
        dir_checkpoint = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'../models/{run_name}/')

    if hard_patches == True:
        hard_patches_percentile = 0.25
    else:
        hard_patches_percentile = 0

    train_dataset = BasicDataset(dir_img, split="train", pattern=pattern, remosaic=remosaic, iso=iso, patch_size=patchsz, patch_num_per_image=patchnum, max_trdata=trimages, hard_patches_percentile=hard_patches_percentile, dropout = dropout)
    val_dataset = BasicDataset(dir_img, split="val", pattern=pattern, remosaic=remosaic, iso=iso, patch_size=patchsz, patch_num_per_image=1, max_trdata=0, hard_patches_percentile=hard_patches_percentile)
    
    n_train = len(train_dataset)
    n_val = len(val_dataset)

    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)#setting batch size smaller to fit on single gpu

    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs} epochs
        Batch size:      {batch_size}
        Patch size:      {patchsz} x {patchsz}
        Patches/image:   {patchnum}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Validation Frq.: {validationFrequency}
        Checkpoints:     {save_cp}
        Device:          {device}
    ''')


    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, lrdp, gamma=lrdf, last_epoch=-1)

    reconstruction_loss = None

    job_info_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'../jobs/{args.job_id}.pkl')
    #load in job file details for vector jobs
    if os.path.exists(job_info_path):
        with open(job_info_path, 'rb') as fp:
            job_info_loaded = pickle.load(fp)
            args.load = job_info_loaded["last_checkpoint"]
            best_psnr = job_info_loaded["best_psnr"]
            if (torch.isinf(best_psnr)):
                best_psnr = 0
                print("LOADED INF PSNR")
            epochs_done = job_info_loaded["epochs_done"]
    else:
        best_psnr = 0
        epochs_done = 0
        
    
    if args.load:
        checkpoint = torch.load(args.load, map_location=device)

        net.load_state_dict(checkpoint['net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        logging.info(f'Model loaded from {args.load}')

    wandb.watch(net, log_freq=100)

    for epoch in range(epochs_done, epochs):
        net.train()
        epoch_loss = 0
        loss_single = None
        with tqdm(total=n_train*patchnum, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs_ = batch['image']
                gt_ = batch['gt']
                assert imgs_.shape[1] == net.module.in_channels * patchnum, \
                    f'Network has been defined with {net.module.in_channels} input channels, ' \
                    f'but loaded training images have {imgs_.shape[1] / patchnum} channels. Please check that ' \
                    'the images are loaded correctly.'

                assert gt_.shape[1] == net.module.out_channels * patchnum, \
                    f'Network has been defined with {net.module.out_channels} input channels, ' \
                    f'but loaded GT images have {gt_.shape[1] / patchnum} channels. Please check that ' \
                    'the images are loaded correctly.'

                for j in range(patchnum):
                    imgs = imgs_[:, (j * net.module.in_channels): net.module.in_channels + (j * net.module.in_channels), :, :]
                    gt = gt_[:, (j * net.module.out_channels): net.module.out_channels + (j * net.module.out_channels), :, :]

        
                    net_out = net(imgs)
                    single_pred, quad_pred, nona_pred = net_out
                    gt = gt.to(device=device, dtype=torch.float32)
                
                    loss_single = mse_loss.compute(single_pred, gt)
                    loss_quad = mse_loss.compute(quad_pred, gt)
                    loss_nona = mse_loss.compute(nona_pred, gt)

                   
                    loss = (loss_single+loss_quad +loss_nona)/3
           
                    epoch_loss += loss.item()

                    log_dict = {'Loss/train': loss.item(), "Global Step": global_step, "Epoch": epoch}

                    if reconstruction_loss is not None:
                        log_dict["Reconstruction loss"] = reconstruction_loss
                    
                    
                    if loss_single is not None:
                        log_dict["Single Loss"] = loss_single
                        log_dict["Quad Loss"] = loss_quad
                        log_dict["Nona Loss"] = loss_nona
                    
                    wandb.log(log_dict)
                    pbar.set_postfix(**{'loss (batch)': loss.item()})
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    pbar.update(batch_size) #batch_size patches seen
                    global_step += 1
                    
        #validation
        if (epoch + 1) % validationFrequency == 0:
            
            outputs= vald_net(net, model_name, val_loader, device, out_imgs_dir=odir, crop_outer_pixels=2, epoch=epoch, discriminator=discriminator)
            wandb.log(outputs)
            average_val_psnr = (outputs["PSNR Val"] + outputs["PSNR Quad"] + outputs["PSNR Nona"])/3

            average_val_psnr = average_val_psnr.cpu() #prevents a bug from checkpoint loading

            if(average_val_psnr > best_psnr):
                save_checkpoint(f'best.pth', dir_checkpoint, net, optimizer, scheduler, None, None, scheduler)
                best_psnr = average_val_psnr
                logging.info(f'Saved new best model at PSNR: {best_psnr}!')
          

        scheduler.step()

        #checkpoint saving
        if save_cp and (epoch + 1) % chkpointperiod == 0:
            save_checkpoint(f'epoch{epoch + 1}.pth', dir_checkpoint, net, optimizer, scheduler, None, None, scheduler)
            
            logging.info(f'Checkpoint {epoch + 1} saved!')

            #keep only 2 checkpoints at a time
            last_checkpoint_num = epoch + 1 - 2*chkpointperiod
            previous_checkpoint = dir_checkpoint + f'epoch{last_checkpoint_num}.pth'

            if (os.path.exists(previous_checkpoint)):
                os.remove(previous_checkpoint)
                logging.info(f'Checkpoint {last_checkpoint_num} deleted!')

        
        #do my job tracking
        if job_id is not None:
            job_info = {"last_checkpoint":dir_checkpoint + f'epoch{epoch + 1}.pth',
                                "wandb_run_id": wandb.run.id,
                                "best_psnr": best_psnr,
                                "epochs_done": epoch+1}
            if not os.path.exists(os.path.dirname(job_info_path)):
                os.makedirs(os.path.dirname(job_info_path), exist_ok=True)
                logging.info('Created Job Info directory')
            with open(job_info_path, 'wb') as fp:
                pickle.dump(job_info, fp)
                print(f'Saved Job Info at Epoch {epoch+1}')
    
    save_checkpoint(f'net.pth', dir_checkpoint, net, optimizer, scheduler, None, None, scheduler)

    logging.info('Saved trained model!')
    logging.info('End of training')


def vald_net(net, name, loader, device, plot_images=False, out_imgs_dir=None,crop_outer_pixels = 0, epoch = 0, discriminator=None):
    """Evaluation using MAE"""
    net.eval()
    quad_pred = None
    nona_pred = None
    bilinear_in = None

    outputs = {}

    def log(key, value):
        if (key in outputs):
            outputs[key] += value
        else:
            outputs[key] = value

    with tqdm(total=len(loader), desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs_ = batch['image']
            gt_ = batch['gt']
            patchnum = 1
            assert imgs_.shape[1] == net.module.in_channels * patchnum, \
                    f'Network has been defined with {net.module.in_channels} input channels, ' \
                    f'but loaded training images have {imgs_.shape[1] / patchnum} channels. Please check that ' \
                    'the images are loaded correctly.'

            assert gt_.shape[1] == net.module.out_channels * patchnum, \
                f'Network has been defined with {net.module.out_channels} input channels, ' \
                f'but loaded GT images have {gt_.shape[1] / patchnum} channels. Please check that ' \
                'the images are loaded correctly.'

            imgs = imgs_[:, :, :, :]
            gt = gt_[:, :, :, :]

            gt = gt.to(device=device, dtype=torch.float32)


            with torch.no_grad():
        
                net_out = net(imgs)
                
                imgs_pred, quad_pred, nona_pred = net_out
                quad_pred = output_rescaling_quantization(quad_pred)
                nona_pred = output_rescaling_quantization(nona_pred)
                imgs_pred = output_rescaling_quantization(imgs_pred)
                gt = output_rescaling_quantization(gt)

                if crop_outer_pixels != 0:
                    imgs = imgs[:, :, crop_outer_pixels:-crop_outer_pixels, crop_outer_pixels:-crop_outer_pixels]
                    imgs_pred = imgs_pred[:, :, crop_outer_pixels:-crop_outer_pixels, crop_outer_pixels:-crop_outer_pixels]
                    if bilinear_in is not None:
                        bilinear_in = bilinear_in[:, :, crop_outer_pixels:-crop_outer_pixels, crop_outer_pixels:-crop_outer_pixels]
                    gt = gt[:, :, crop_outer_pixels:-crop_outer_pixels, crop_outer_pixels:-crop_outer_pixels]
                    if nona_pred is not None:
                        nona_pred = nona_pred[:, :, crop_outer_pixels:-crop_outer_pixels, crop_outer_pixels:-crop_outer_pixels]
                    if quad_pred is not None:
                        quad_pred = quad_pred[:, :, crop_outer_pixels:-crop_outer_pixels, crop_outer_pixels:-crop_outer_pixels]
                    
                batch_size, loss = mse_loss_sum_across_batch.compute(imgs_pred, gt)
                _, psnr_batch, psnrs_per_image = psnr_per_image.compute(imgs_pred, gt)
                #compute the ssim for the batch
                _, ssim_batch, ssims_per_image = ssim_per_image.compute(imgs_pred, gt)

                _, deltae_batch, deltaes_per_image, imgs_pred_srgb, gts_srgb = deltae_per_image.compute(imgs_pred, gt)


                log("DeltaE Val", deltae_batch)
                log("SSIM Val", ssim_batch)
                log("PSNR Val", psnr_batch)
                log("MSE Val", loss)
                log("Number of items", batch_size)

                if quad_pred is not None:
                    _, quad_loss = mse_loss_sum_across_batch.compute(quad_pred, gt)
                    _, quad_psnr_batch, quad_psnrs_per_image = psnr_per_image.compute(quad_pred, gt)
                    #compute the ssim for the batch
                    _, quad_ssim_batch, quad_ssims_per_image = ssim_per_image.compute(quad_pred, gt)
                    _, quad_deltae_batch, quad_deltaes_per_image, quad_pred_srgb, gts_srgb = deltae_per_image.compute(quad_pred, gt)
                    log("DeltaE Quad", quad_deltae_batch)
                    log("SSIM Quad", quad_ssim_batch)
                    log("MSE Quad", quad_loss)
                    log("PSNR Quad", quad_psnr_batch)

                    _, nona_loss = mse_loss_sum_across_batch.compute(nona_pred, gt)
                    _, nona_psnr_batch, nona_psnrs_per_image = psnr_per_image.compute(nona_pred, gt)
                    #compute the ssim for the batch
                    _, nona_ssim_batch, nona_ssims_per_image = ssim_per_image.compute(nona_pred, gt)
                    _, nona_deltae_batch, nona_deltaes_per_image,nona_pred_srgb, gts_srgb = deltae_per_image.compute(nona_pred, gt)
                    log("DeltaE Nona", nona_deltae_batch)
                    log("SSIM Nona", nona_ssim_batch)
                    log("MSE Nona", nona_loss)
                    log("PSNR Nona", nona_psnr_batch)

                if plot_images:
                    log_dict={}
                    if quad_pred is not None:
                        log_dict["Val Quad Predicted"] = generate_output_images(out_imgs_dir+"quad", quad_pred, quad_psnrs_per_image, batch["img_file"], )
                    if nona_pred is not None:
                        log_dict["Val Nona Predicted"] = generate_output_images(out_imgs_dir+"nona", nona_pred, nona_psnrs_per_image, batch["img_file"])
                    log_dict["Val Predicted"] = generate_output_images(out_imgs_dir, imgs_pred, psnrs_per_image, batch["img_file"])
                    log_dict["Val GT"] = generate_output_images(out_imgs_dir+"gt", gt, np.ones(psnrs_per_image.shape[0]), batch["img_file"])
                    wandb.log(log_dict)

            pbar.update(1)
            
 
    #normalize all the outputs by the number of items except number of items
    for key in outputs.keys():
        if key != "Number of items":
            outputs[key] = outputs[key]/outputs["Number of items"]

    #remove the number of items key
    outputs.pop("Number of items")

    #set network back to train mode
    net.train()

    return outputs


if __name__ == '__main__':
    #set random seeds
    torch.manual_seed(0)
    np.random.seed(0)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    args = get_args()
    args = set_job_id(args)
    
    gpus_chosen, device = get_device(args)
    net, discriminator, teacher = get_model(args,device)

    logging.info(f'Using model {args.model}')
    net.to(device)

    net = torch.nn.DataParallel(net, device_ids=gpus_chosen)


    if teacher is not None:
        for i in range(len(teacher)):
            teacher[i].to(device)
            teacher[i] = torch.nn.DataParallel(teacher[i], device_ids=gpus_chosen)


    if discriminator is not None:
        discriminator = discriminator.to(device)
        discriminator = torch.nn.DataParallel(discriminator, device_ids=gpus_chosen)

    run_name = setup_wandb_config(args, net, 0, 0)
    
    if not args.test:

        delete_old_wandb_files(args.owner_name)

        #restart wandb progress if stuff got stopped in the middle
        job_info_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'../jobs/{args.job_id}.pkl')
        if (os.path.exists(job_info_path)):
            resume="must"
            with open(job_info_path, 'rb') as fp:
                job_info_loaded = pickle.load(fp)
                run_name = job_info_loaded["wandb_run_id"]
        else:
            resume = False

        wandb.init(entity=args.wandb_entity,
            id = run_name,
        project="SSDemosaic",
        config = wandb.config,
        name = run_name,
        notes="",
        resume=resume)

        odir = os.path.join(args.odir, run_name)

        logging.info('Training of Demosaicing Algorithms')

        train_net(net=net,
                  model_name = args.model,
                    discriminator=discriminator,
                    pattern = args.pattern,
                    remosaic = args.remosaic,
                    iso = args.iso,
                    epochs=args.epochs,
                    batch_size=args.batchsize,
                    lr=args.lr,
                    lrdf=args.lrdf,
                    lrdp=args.lrdp,
                    device=device,
                    chkpointperiod=args.chkpointperiod,
                    trimages=args.trimages,
                    validationFrequency=args.val_frq,
                    patchsz=args.patchsz,
                    patchnum=args.patchnum,
                    dir_img=args.patchdir,
                    num_workers=args.workers,
                    run_name = run_name,
                    job_id = args.job_id,
                    hard_patches = args.hard_patches,
                    dropout = args.dropout,
                    odir=odir,)
                    
    else:
        wandb.init(project="SSDemosaic_ValTest",
        config = wandb.config,
        name = run_name,
        notes="")

        odir = os.path.join(args.odir, run_name)
        os.makedirs(odir, exist_ok=True)
                
        logging.info('Testing of Demosaicing Algorithms')

        assert args.load, "specify the model to load" 

        checkpoint = torch.load(args.load, map_location=device)
        net.load_state_dict(checkpoint['net_state_dict'])

        dropouts = [0.00, 0.01]
           

        for percentile in dropouts:
            test_dataset = BasicDataset(args.patchdir, split="test", pattern=args.pattern, remosaic=args.remosaic, patch_size=args.patchsz, iso=args.iso, patch_num_per_image=args.patchnum, hard_patches_percentile=0.25, dropout=percentile, noisy_viz = args.noisy_viz,mask_interpolation=args.mask_interpolation)
            test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False)
            outputs = vald_net(net, 
            args.model, test_loader, device, plot_images=args.plot_images, out_imgs_dir=odir,crop_outer_pixels = 2)
            
            #add dropout to all the keys
            for key in list(outputs.keys()):
                outputs[f"Drop{percentile} {key}"] = outputs.pop(key)
            wandb.log(outputs)
      

        test_full_dataset = BasicDataset(args.fulldir, split="test_full_size", pattern=args.pattern, remosaic=args.remosaic, iso=args.iso, patch_size=args.patchsz, patch_num_per_image=args.patchnum)
        test_full_loader = DataLoader(test_full_dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True)

        outputs = vald_net(net, args.model, test_full_loader, device, out_imgs_dir=odir,plot_images=args.plot_images, crop_outer_pixels = 2)

        #add "Full" to all the keys
        for key in list(outputs.keys()):
            outputs[f"Full {key}"] = outputs.pop(key)
        
        wandb.log(outputs)