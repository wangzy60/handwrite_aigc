import torch.distributed
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import argparse
import cv2
import numpy as np
import os
import math
from glob import glob
from torchvision import transforms
from PIL import Image
import time
from tqdm import tqdm
import logging
import datetime
import random
import math
import matplotlib.pyplot as plt
from pathlib import Path

# from model.unet import Unet2 as Unet
from model.vae import KL_VAE as VAE
from model.my_lpips_loss import LPIPS 
from model.dataset import VAE_Dataset
from utils.ddpm_schedule import add_noise
from utils.utils import clip_vit_base_patch32_model_infer
from utils.utils import get_device, get_image_transformer, detransform_tensor2image, ldm_load_model
from utils.ddpm_schedule import alpha_t_list,alpha_t_bar_list,beta_t_list,beta_t_bar_list

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision.models import inception_v3
import lpips
from scipy.linalg import sqrtm
import sys


# 针对旧版NumPy的兼容性补丁
if np.__version__ < "1.26.0":
    class DummyCoreModule:
        __path__ = []  # 使模块表现为包
        __all__ = []   # 避免导入错误
    
    # 创建虚拟模块链
    sys.modules["numpy._core"] = DummyCoreModule()
    sys.modules["numpy._core._multiarray_umath"] = DummyCoreModule()
    
    # 将实际功能重定向到旧版位置
    import numpy.core as actual_core
    sys.modules["numpy._core"] = actual_core
    sys.modules["numpy._core._multiarray_umath"] = actual_core._multiarray_umath

#绘制最近一个epoch的损失曲线:done
#学习率调度:done
#训练恢复:done
#恢复训练后减少恢复步数需要的时间
#FID损失计算
#DDIM采样实现
#IS损失计算        


def get_unet(args):
    if args.input_image_dims == 4 and args.output_image_dims == 4:
        from model.unet import Unet_With_Text_Condition as Unet
    elif args.input_image_dims == 3 and args.output_image_dims == 3:
        # from model.unet import Unet_Without_Condition as Unet
        from model.unet import Unet3 as Unet
    else:
        raise ValueError(f"获得Unet模型失败，args.input_image_dims: {args.input_image_dims}和args.output_image_dims: {args.output_image_dims} 必须相等，且只能为3或4")
    logging.info(f"使用的模型为{Unet}")
    return Unet

def get_ddpm_dataset(args):
    if args.input_image_dims == 4 and args.output_image_dims == 4:
        from model.dataset import LDM_Flickr30K_CLIP_Dataset as DDPM_Dataset
    elif args.input_image_dims == 3 and args.output_image_dims == 3:
        from model.dataset import DDPM_Flickr30K_Dataset as DDPM_Dataset
    else:
        raise ValueError(f"获得DDPM数据集类型失败，args.input_image_dims: {args.input_image_dims}和args.output_image_dims: {args.output_image_dims} 必须相等，且只能为3或4")
    logging.info(f"使用的模型为{DDPM_Dataset}")
    return DDPM_Dataset


def print_on_rank0(anything):
    rank = int(os.environ["RANK"])
    if rank == 0:
        print(anything)


def get_inception_v3_features(image_path_list, device, batch_size):
    transform = transforms.Compose([transforms.Resize((299, 299)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    model = inception_v3(pretrained=True)
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()
    loader = torch.utils.data.DataLoader(image_path_list, batch_size=batch_size, shuffle=False)
    features = []
    with torch.no_grad():
        for images in tqdm(loader):
            images = [transform(Image.open(image)).unsqueeze(0).to(device) for image in images]
            images = torch.cat(images, dim=0)
            feature = model(images)
            features.append(feature)
    features = torch.cat(features, dim=0)
    return features.cpu().numpy()
    

def computer_fid(real_features, fake_features):
    mean_real = np.mean(real_features, axis=0)
    mean_fake = np.mean(real_features, axis=0)
    conv_real = np.cov(real_features, rowvar=True)
    conv_fake = np.cov(fake_features, rowvar=True)
    conv_mean = sqrtm(conv_real @ conv_fake)
    if not np.isfinite(conv_mean).all():
        epsilon = np.eye(conv_real.shape[0]) * 1e-6
        conv_mean = sqrtm((conv_real + epsilon) @ (conv_fake + epsilon))
    conv_mean = conv_mean.real
    fid = np.sum((mean_real - mean_fake)**2) + np.trace(conv_real + conv_fake - 2 * conv_mean)
    return float(fid)

def get_fid_loss(real_image_path_list, fake_image_path_list, device, batch_size):
    print("---计算真实图片的特征向量---")
    real_features = get_inception_v3_features(real_image_path_list, device, batch_size)
    print("---计算生成图片的特征向量---")
    fake_features = get_inception_v3_features(fake_image_path_list, device, batch_size)
    return computer_fid(real_features, fake_features)


def test_computer_fid_loss(args):
    test_image_num = 5000
    image_path = os.path.join(os.path.abspath(args.train_image_fold), "*.jpg")
    image_path_list = glob(image_path)
    real_image_path_list = image_path_list[:test_image_num]
    fake_image_path_list = image_path_list[test_image_num:2*test_image_num]
    device = get_device(args.device)
    batch_size = args.batch_size
    
    print("---计算真实图片和生成图片的FID损失---")
    # fid_loss = get_fid_loss(real_image_path_list, fake_image_path_list, device, batch_size)
    from pytorch_fid import fid_score
    fid_loss = fid_score.calculate_fid_given_paths(["/shared_file/hand_write_aigc/dataset/test1", "/shared_file/hand_write_aigc/dataset/test2"], batch_size=batch_size, device=device, dims=2048)
    print(f"FID损失为：{fid_loss:.4f}")


class DDPM1_Dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, shift_func, noise_func, total_steps = 1000, transform = None):
        super().__init__()
        self.image_paths = glob(os.path.join(image_dir, '*.jpg'))
        self.transform = transform
        self.total_steps = total_steps
        self.shift_func = shift_func
        self.noise_func = noise_func

    def __len__(self):
        return len(self.image_paths)
    
    def add_noise(self, x, t, verbose=False):
        return add_noise(x, t, self.shift_func, self.noise_func, verbose=verbose)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        t = torch.randint(1, self.total_steps + 1, (1,))  #这里需要用total_steps+1，因为torch.randint不包含最后一个数
        img_with_noisy, noise_img, noise = self.add_noise(image, t)
        return img_with_noisy, t, noise


def show_image_with_noise(image_path, max_t, iter_gap, shift_func, noise_func, save_image= True, verbose = False, image_save_path = None, noise_save_path = None):
    img_with_noisy_list = []
    noise_img_list = []
    # t_list = []
    iter_nums = math.ceil(max_t/iter_gap)
    img = Image.open(image_path)
    transformer = get_image_transformer(img.width)
    img_tansformered = transformer(img)
    for i in range(0, iter_nums+1):
        t = i*iter_gap if i*iter_gap < max_t-1 else max_t-1
        img_with_noisy, noise_img, _ = add_noise(img_tansformered, t, shift_func, noise_func,verbose=verbose)
        img_with_noisy = detransform_tensor2image(img_with_noisy)
        noise_img = detransform_tensor2image(noise_img)
        img_with_noisy_list.append(img_with_noisy)
        noise_img_list.append(noise_img)
        # t_list.append(t)
    if save_image:
        img_with_noisy_concat = concatenate_images_horizontally(img_with_noisy_list)
        noise_img_concat = concatenate_images_horizontally(noise_img_list)
        if image_save_path is None:
            image_save_path = os.path.join("./noise_img", os.path.abspath(image_path).split('celebA_HQ/')[1].replace('.jpg', '_with_noise_concat.jpg'))
            noise_save_path = os.path.join("./noise_img", os.path.abspath(image_path).split('celebA_HQ/')[1].replace('.jpg', '_noise.jpg'))
        if not os.path.exists(os.path.dirname(image_save_path)):
            os.makedirs(os.path.dirname(image_save_path))
        img_with_noisy_concat.save(image_save_path)
        noise_img_concat.save(noise_save_path)

    return img_with_noisy_list, noise_img_list


def show_noise_image():
    image_path = "./dataset/celebA_HQ/data128x128/00003.jpg"
    max_t = 1000
    iter_gap = 5
    return show_image_with_noise(image_path, max_t, iter_gap, alpha_t_bar_list, beta_t_bar_list, verbose=True)

def plot_training_loss(save_path, losses):
    """
    绘制深度学习网络训练过程中所有batch的损失值，并控制x轴的显示。

    参数:
    save_path (str): 图片存储的地址。
    losses (list of float): 每个epoch和batch对应的损失值。
    """
    # 创建一个新的图形
    plt.figure(figsize=(10, 6))

    # 遍历所有损失值
    plt.plot(list(range(len(losses))), losses, 'r')

    # 设置x轴刻度和标签
    plt.xlabel('Epoch/Batch')

    # 设置图形的标题和y轴标签
    plt.title('Training Loss Over All Batches')
    plt.ylabel('Loss')
    # plt.legend()

    # 保存图形为图片
    plt.savefig(save_path)
    plt.close()


def ddpm1_train(args):
    ckpt_save_fold = os.path.join(args.project_path, args.ckpt_save_fold)
    loss_img_save_path = os.path.join(args.project_path, 'loss_img.png')
    train_mse_loss_mode = args.train_mse_loss_mode
    if not os.path.exists(ckpt_save_fold):
        os.mkdir(ckpt_save_fold)
    device = get_device(args.device)
    train_image_fold = args.train_image_fold
    total_steps = args.total_steps
    net = get_unet(args)
    model = net(args).to(device)
    logging.info(model)
    image_transformer = get_image_transformer(args.input_image_size)
    celebA_HQ_dataset = DDPM1_Dataset(train_image_fold, alpha_t_bar_list, beta_t_bar_list, total_steps=total_steps, transform=image_transformer)
    dataloader = torch.utils.data.DataLoader(celebA_HQ_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    best_loss = 1e8
    total_batchs = len(dataloader)
    batch_list = []
    epoch_list = []
    loss_list  = []
    for epoch in range(args.epochs):
        for batch_index, batch in enumerate(dataloader):
            images_add_noise, time_step, noises = batch
            images_add_noise = images_add_noise.to(device)
            time_step = time_step.to(device)
            noises = noises.to(device)
            optimizer.zero_grad()
            predict_noise = model(images_add_noise, time_step)
            if train_mse_loss_mode == 'sum':
                loss = F.mse_loss(predict_noise, noises, reduction=train_mse_loss_mode)/len(images_add_noise)
            elif train_mse_loss_mode == 'mean':
                loss = F.mse_loss(predict_noise, noises, reduction=train_mse_loss_mode)
            loss.backward()
            optimizer.step()
            logging.info(f"epoch: {epoch}, batch: {batch_index}/{total_batchs}, loss: {loss.item()}, best_loss: {best_loss:.6f}")
            if best_loss > loss.item():
                best_loss = loss.item()
                if epoch >= 5:
                    torch.save({'state_dict': model.state_dict(),
                                "epoch": epoch,
                                "batch": batch_index,
                                "loss": loss.item()}, 
                                os.path.join(ckpt_save_fold, f"ddpm1_best_model.pth"))
            if (epoch % int(args.save_every_epochs) == 0) and (batch_index == 0):
                torch.save({'state_dict': model.state_dict(),
                            "epoch": epoch,
                            "batch": batch_index,
                            "loss": loss.item()}, 
                           os.path.join(ckpt_save_fold, f"ddpm1_epoch_{epoch}_batch_{batch_index}_loss_{loss.item():.4f}.pth"))
            #绘制损失曲线
            epoch_list.append(epoch)
            batch_list.append(batch_index)
            loss_list.append(loss.item())
        #每个epoch保存一次
        plot_training_loss(loss_img_save_path, epoch_list, batch_list, loss_list)
            

def concatenate_images_horizontally(image_numpy_list):
    assert len(image_numpy_list) > 0, "image_numpy_list不能为空"
    image_num = len(image_numpy_list)

    img_width = image_numpy_list[0].width
    total_width = int(img_width * image_num)
    max_height = image_numpy_list[0].height

    new_image = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for img in image_numpy_list:
        new_image.paste(img, (x_offset, 0))
        x_offset += img_width
    return new_image


def concatenate_images_vertical(image_numpy_list):
    assert len(image_numpy_list) > 0, "image_numpy_list不能为空"
    image_num = len(image_numpy_list)

    img_height = image_numpy_list[0].height
    total_height = int(img_height * image_num)
    total_width = image_numpy_list[0].width

    new_image = Image.new('RGB', (total_width, total_height))

    x_offset = 0
    for img in image_numpy_list:
        new_image.paste(img, (0, x_offset))
        x_offset += img_height
    return new_image


def ddpm_validation(model, args, epoch_index):
    #支持同时推理多张图片
    device = next(model.parameters()).device
    model.eval()
    total_steps = int(args.total_steps)
    batch_size = 10
    ddpm_prompt_list = args.ddpm_prompt_list
    denoise_steps_gap = 1
    inference_image_save_fold = args.project_path
    with torch.no_grad():
        logging.info(f"第{epoch_index}轮训练开始推理...")
        x_t = torch.randn(batch_size, 3, args.input_image_size, args.input_image_size).to(device)
        if ddpm_prompt_list:
            embeddings = clip_vit_base_patch32_model_infer(ddpm_prompt_list, device)
        image_name = f"train_{epoch_index}_epochs_infer.jpg"
        image_save_path = os.path.join(inference_image_save_fold, image_name)
        time_schedule = list(range(total_steps-1, -1, -1 * denoise_steps_gap))

        #ddpm
        for t in tqdm(time_schedule):
            time_step = torch.tensor([t]*batch_size).to(device)
            if ddpm_prompt_list:
                predict_noise = model(x_t, time_step, embeddings)
            else:
                predict_noise = model(x_t, time_step)
            a_t = alpha_t_list[t]
            b_t = beta_t_list[t]
            b_t_bar = beta_t_bar_list[t]
            b_t_minus_one_bar = beta_t_bar_list[t-1] if t - 1 >= 0 else 0
            sigma = b_t * b_t_minus_one_bar  / b_t_bar
            # sigma = b_t
            x_t = 1 / a_t * (x_t - b_t**2 / b_t_bar * predict_noise) + sigma * torch.randn_like(x_t)
  
        
        # ##ddim
        # if time_schedule[-1] != 0:
        #     time_schedule.append(0)
        # for i in tqdm(range(len(time_schedule))):
        #     t = time_schedule[i]
        #     time_step = torch.tensor([t]*batch_size).to(device)
        #     # predict_noise = model(x_t, time_step, embeddings)
        #     predict_noise = model(x_t, time_step)
        #     p_t = time_schedule[i+1] if i+1 < len(time_schedule) else None
        #     a_t = float(alpha_t_bar_list[t])
        #     a_pt = float(alpha_t_bar_list[p_t] if i+1 < len(time_schedule) else 1)
        #     b_t = float(beta_t_bar_list[t])
        #     b_pt = float(beta_t_bar_list[p_t] if i+1 < len(time_schedule) else 0)
        #     sigma = torch.sqrt(torch.tensor(1.0) - (a_t**2)/(a_pt**2)) * b_pt  / b_t
        #     x_t = (a_pt / a_t) * (x_t + (a_t*torch.sqrt(b_pt**2 - sigma**2)/a_pt - b_t) * predict_noise) + torch.randn_like(x_t) * sigma

        #     #保存过程图片
        #     if (inference_show_denoise_image_every_n_steps is not None) and (t < image_save_taggle):
        #         image_save_taggle -= inference_show_denoise_image_every_n_steps
        #         image_list.append(detransform_tensor2image(x_t.detach().cpu()))        

        #保存最终图片
        image = detransform_tensor2image(x_t.detach().cpu())
        concatenated_image = concatenate_images_horizontally(image)
        concatenated_image.save(image_save_path)
    del x_t, time_step
    if ddpm_prompt_list:
        del embeddings
    logging.info(f"第{epoch_index}轮训练推理成功，图片保存至{image_save_path}")
    model.train()


def ddpm_inference(model, args):
    #支持同时推理多张图片
    device = next(model.parameters()).device
    total_steps = int(args.total_steps)
    model.eval()
    batch_size = int(args.batch_size)
    ddpm_prompt_list = args.ddpm_prompt_list
    denoise_steps_gap = int(args.denoise_steps_gap)
    inference_image_save_fold = args.inference_image_save_fold
    inference_show_denoise_image_every_n_steps = args.inference_show_denoise_image_every_n_steps
    with torch.no_grad():
        print("开始推理...")
        image_list = []
        image_save_taggle = total_steps
        x_t = torch.randn(batch_size, 3, args.input_image_size, args.input_image_size).to(device)
        if ddpm_prompt_list:
            embeddings = clip_vit_base_patch32_model_infer(ddpm_prompt_list, device)
        image_name = f"infer_gap_{denoise_steps_gap}.jpg"
        image_save_path = os.path.join(inference_image_save_fold, image_name)
        time_schedule = list(range(total_steps-1, -1, -1 * denoise_steps_gap))

        #ddpm
        for t in tqdm(time_schedule):
            time_step = torch.tensor([t]*batch_size).to(device)
            if ddpm_prompt_list:
                predict_noise = model(x_t, time_step, embeddings)
            else:
                predict_noise = model(x_t, time_step)
            a_t = alpha_t_list[t]
            b_t = beta_t_list[t]
            b_t_bar = beta_t_bar_list[t]
            b_t_minus_one_bar = beta_t_bar_list[t-1] if t - 1 >= 0 else 0
            sigma = b_t * b_t_minus_one_bar  / b_t_bar
            # sigma = b_t
            x_t = 1 / a_t * (x_t - b_t**2 / b_t_bar * predict_noise) + sigma * torch.randn_like(x_t)

            #保存过程图片
            if (inference_show_denoise_image_every_n_steps is not None) and (t < image_save_taggle):
                image_save_taggle -= inference_show_denoise_image_every_n_steps
                image_list.append(detransform_tensor2image(x_t.detach().cpu()))  
        
        # ##ddim
        # if time_schedule[-1] != 0:
        #     time_schedule.append(0)
        # for i in tqdm(range(len(time_schedule))):
        #     t = time_schedule[i]
        #     time_step = torch.tensor([t]*batch_size).to(device)
        #     # predict_noise = model(x_t, time_step, embeddings)
        #     predict_noise = model(x_t, time_step)
        #     p_t = time_schedule[i+1] if i+1 < len(time_schedule) else None
        #     a_t = float(alpha_t_list[t])
        #     b_t = float(beta_t_list[t])
        #     b_t_bar = float(beta_t_bar_list[t])
        #     b_pt_bar = float(beta_t_bar_list[p_t] if i+1 < len(time_schedule) else 0)
        #     sigma = b_t * b_pt_bar / b_t_bar
        #     x_t = (x_t - (b_t_bar - a_t * ((b_pt_bar**2 - sigma**2) ** 0.5)) * predict_noise) / a_t + torch.randn_like(x_t) * sigma

            # #保存过程图片
            # if (inference_show_denoise_image_every_n_steps is not None) and (t < image_save_taggle):
            #     image_save_taggle -= inference_show_denoise_image_every_n_steps
            #     image_list.append(detransform_tensor2image(x_t.detach().cpu()))        

        #保存最终图片
        image = detransform_tensor2image(x_t.detach().cpu())
        concatenated_image = concatenate_images_horizontally(image)
        concatenated_image.save(image_save_path)
        
        #保存过程图片
        if (inference_show_denoise_image_every_n_steps is not None):
            image_list.append(image)
            concatenated_image = [concatenate_images_vertical(image) for image in image_list]
            concatenated_image = concatenate_images_horizontally(concatenated_image)
            concat_image_save_path = os.path.join(inference_image_save_fold, f"infer_gap_{denoise_steps_gap}_every_{inference_show_denoise_image_every_n_steps}_steps.png")
            concatenated_image.save(concat_image_save_path)



def ldm_inference(args, vae = VAE):
    #支持同时推理多张图片
    unet = get_unet(args)
    device = get_device(args.device)
    total_steps = int(args.total_steps)
    unet_model_path = args.unet_ckpt_path
    vae_model_path = args.vae_ckpt_path
    print("unet_model_path: ", unet_model_path)
    print("vae_model_path: ", vae_model_path)

    unet_model = unet(args)
    unet_model = ldm_load_model(unet_model_path, unet_model).to(device)
    unet_model.eval()
    
    vae_model = vae(args)
    vae_decoder_model = ldm_load_model(vae_model_path, vae_model).decoder.to(device)
    vae_decoder_model.eval()
    
    batch_size = int(args.batch_size)
    ddpm_prompt_list = args.ddpm_prompt_list
    denoise_steps_gap = int(args.denoise_steps_gap)
    inference_image_save_fold = args.inference_image_save_fold
    inference_show_denoise_image_every_n_steps = args.inference_show_denoise_image_every_n_steps
    with torch.no_grad():
        print("开始推理...")
        image_list = []
        image_save_taggle = total_steps
        x_t = torch.randn(batch_size, 4, int(args.input_image_size), int(args.input_image_size)).to(device)
        print("ddpm_prompt_list: " , ddpm_prompt_list)
        if not ddpm_prompt_list:
            ddpm_prompt_list = [""] * batch_size
        embeddings = clip_vit_base_patch32_model_infer(ddpm_prompt_list, device)            
        image_name = f"infer_gap_{denoise_steps_gap}.jpg"
        image_save_path = os.path.join(inference_image_save_fold, image_name)
        time_schedule = list(range(total_steps-1, -1, -1 * denoise_steps_gap))

        #ddpm
        for t in tqdm(time_schedule):
            time_step = torch.tensor([t]*batch_size).to(device)
            predict_noise = unet_model(x_t, time_step, embeddings)
            a_t = alpha_t_list[t]
            b_t = beta_t_list[t]
            b_t_bar = beta_t_bar_list[t]
            b_t_minus_one_bar = beta_t_bar_list[t-1] if t - 1 >= 0 else 0
            sigma = b_t * b_t_minus_one_bar  / b_t_bar
            # sigma = b_t
            x_t = 1 / a_t * (x_t - b_t**2 / b_t_bar * predict_noise) + sigma * torch.randn_like(x_t)


        
        # ##ddim
        # if time_schedule[-1] != 0:
        #     time_schedule.append(0)
        # for i in tqdm(range(len(time_schedule))):
        #     t = time_schedule[i]
        #     time_step = torch.tensor([t]*batch_size).to(device)
        #     # predict_noise = model(x_t, time_step, embeddings)
        #     predict_noise = model(x_t, time_step)
        #     p_t = time_schedule[i+1] if i+1 < len(time_schedule) else None
        #     a_t = float(alpha_t_bar_list[t])
        #     a_pt = float(alpha_t_bar_list[p_t] if i+1 < len(time_schedule) else 1)
        #     b_t = float(beta_t_bar_list[t])
        #     b_pt = float(beta_t_bar_list[p_t] if i+1 < len(time_schedule) else 0)
        #     sigma = torch.sqrt(torch.tensor(1.0) - (a_t**2)/(a_pt**2)) * b_pt  / b_t
        #     x_t = (a_pt / a_t) * (x_t + (a_t*torch.sqrt(b_pt**2 - sigma**2)/a_pt - b_t) * predict_noise) + torch.randn_like(x_t) * sigma

        #     #保存过程图片
        #     if (inference_show_denoise_image_every_n_steps is not None) and (t < image_save_taggle):
        #         image_save_taggle -= inference_show_denoise_image_every_n_steps
        #         image_list.append(detransform_tensor2image(x_t.detach().cpu()))        

        # vae decoder
        mu = x_t * 6.506264453510935  # 6.506264453510935是flickr30K图片经过训练好的vae encoder之后输出的标准差，在训练ddpm时，对每个输入数据除以了这个数
        std = 1
        # latent_z = mu + std * torch.randn_like(mu)
        latent_z = mu
        pred_x = vae_decoder_model(latent_z)


        #保存最终图片
        image = detransform_tensor2image(pred_x.detach().cpu())
        concatenated_image = concatenate_images_horizontally(image)
        concatenated_image.save(image_save_path)
        
        # #保存过程图片
        # if (inference_show_denoise_image_every_n_steps is not None):
        #     image_list.append(image)
        #     concatenated_image = [concatenate_images_vertical(image) for image in image_list]
        #     concatenated_image = concatenate_images_horizontally(concatenated_image)
        #     concat_image_save_path = os.path.join(inference_image_save_fold, f"infer_gap_{denoise_steps_gap}_every_{inference_show_denoise_image_every_n_steps}_steps.png")
        #     concatenated_image.save(concat_image_save_path)



def show_demoise_image():
    image_path = "./dataset/celebA_HQ/data128x128/00003.jpg"
    max_t = 1000
    iter_gap = 1
    with torch.no_grad():
        img = Image.open(image_path)
        transformer = get_image_transformer(img.width)
        img_tansformered = transformer(img)
        t = 100
        a_t_bar = alpha_t_bar_list[t]
        b_t_bar = beta_t_bar_list[t]
        img_with_noisy, noise_img, noise = add_noise(img_tansformered, t, alpha_t_bar_list, beta_t_bar_list,verbose=False)
        img_with_noisy_tmp = detransform_tensor2image(img_with_noisy)
        img_with_noisy_new = transformer(img_with_noisy_tmp)
        # print(img_with_noisy_new - img_with_noisy)
        noise_img_tmp = detransform_tensor2image(noise_img)
        noise_img_new = transformer(noise_img_tmp)
        # print(noise_img_new - noise_img)
        x_0_tmp = (1/a_t_bar) *(img_with_noisy_new - noise_img_new)
        x_0 = (1/a_t_bar) *(img_with_noisy - noise_img)
        image_save_path = os.path.join("./noise_img", os.path.abspath(image_path).split('celebA_HQ/')[1].replace('.jpg', '_with_denoise.jpg'))

        #保存最终图片
        image = detransform_tensor2image(x_0.detach().cpu())
        # print(np.array(image)-np.array(img))
        image.save(image_save_path)


def test():
    image_path = "/shared_file/hand_write_aigc/noise_img/data128x128/00003.jpg"
    image_save_path = "/shared_file/hand_write_aigc/noise_img/data128x128/00003_with_denoise.jpg"
    img = Image.open(image_path)
    transformer = get_image_transformer(img.width)
    img_tansformered = transformer(img)
    new_image = detransform_tensor2image(img_tansformered)
    new_image.save(image_save_path)

def get_random_state_dict():
    random_state_dict = {}
    cuda_random_state = [torch.cuda.get_rng_state(index) for index in range(torch.cuda.device_count())]
    torch_rando_state = torch.get_rng_state()
    numpy_random_state = np.random.get_state()
    python_random_state = random.getstate()
    random_state_dict = {"cuda": cuda_random_state,
                         "torch": torch_rando_state,
                         "numpy": numpy_random_state,
                         "python": python_random_state}
    return random_state_dict


def save_checkpoint(epoch:int, 
                    batch_index:int, 
                    global_steps:int, 
                    best_loss_value:float, 
                    model:dict, 
                    optimizer:dict,
                    scheduler:dict,
                    ckpt_save_path: str):
    model_state_dict = model.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    scheduler_state_dict = scheduler.state_dict()
    random_state_dict = get_random_state_dict()
    try:
        ckpt_dict = {"epoch": epoch,
                    "batch": batch_index,
                    "global_steps": global_steps,
                    "best_loss": best_loss_value, 
                    "model_state_dict": model_state_dict,
                    "optimizer_state_dict": optimizer_state_dict,
                    "scheduler_state_dict": scheduler_state_dict,
                    "random_state_dict": random_state_dict}
        torch.save(ckpt_dict, ckpt_save_path)
        logging.info(f"ckpt保存成功，epoch:{epoch} batch:{batch_index} global_steps:{global_steps} best_loss:{best_loss_value}，保存至 {ckpt_save_path}")
    except:
        logging.info(f"ckpt保存失败，epoch:{epoch} batch:{batch_index} global_steps:{global_steps} best_loss:{best_loss_value}，未保存至 {ckpt_save_path}")


def load_state_dict(state_dict, rank):
    for k, v in state_dict.items():
        if torch.is_tensor(v):  #只广播张量数据，且torch.distributed.broadcast只能广播张量数据
            v = v.to(rank)  #确保张量都在GPU设备上，因为torch.distributed.broadcast广播数据时需要数据都在同类型的设备上
            torch.distributed.broadcast(v, src=0)  #src所代表的rank发送v，其他rank接受v并替换自己已有的v的值
            state_dict[k] = v
    return state_dict


def load_checkpoint(ckpt_path, model, optimizer, scheduler, rank):
    #rank 0 加载断点模型
    if rank == 0:
        checkpoint_dict = torch.load(ckpt_path, weights_only=False)
        model_state_dict = checkpoint_dict["model_state_dict"]
        optimizer_state_dict = checkpoint_dict["optimizer_state_dict"]
        scheduler_state_dict = checkpoint_dict["scheduler_state_dict"]
        random_state_dict = [checkpoint_dict["random_state_dict"]]  #使用broadcast_object_list广播数据，需要用list包裹起来，且广播和待广播的列表长度需要一致
        train_state_dict = [{"epoch": checkpoint_dict["epoch"],
                            "best_loss": checkpoint_dict["best_loss"],
                            "global_steps": checkpoint_dict["global_steps"]}]
    #其他节点生产对应的数据格式，数据内容可以随意
    else:
        model_state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict()
        scheduler_state_dict = scheduler.state_dict()
        random_state_dict = [None]
        train_state_dict = [None]
    
    #rank 0 广播所有数据，其他节点接收并替换对应数据
    model_state_dict = load_state_dict(model_state_dict, rank)
    optimizer_state_dict = load_state_dict(optimizer_state_dict, rank)
    scheduler_state_dict = load_state_dict(scheduler_state_dict, rank)
    torch.distributed.broadcast_object_list(random_state_dict, src=0)
    torch.distributed.broadcast_object_list(train_state_dict, src=0)

    #所有节点恢复权重数据，恢复随机状态，恢复训练状态
    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)
    scheduler.load_state_dict(scheduler_state_dict)

    train_state_dict = train_state_dict[0]
    epoch, best_loss, global_steps = train_state_dict["epoch"], train_state_dict["best_loss"], train_state_dict["global_steps"]
    random_state_dict = random_state_dict[0]
    random.setstate(random_state_dict["python"])
    np.random.set_state(random_state_dict["numpy"])
    torch.set_rng_state(random_state_dict["torch"])
    torch.cuda.set_rng_state(random_state_dict["cuda"][rank])

    return epoch, best_loss, global_steps
        

def multi_gpu_ddpm_train(args, save_ckpt = True):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    accumulate_grad_batches = args.accumulate_grad_batches

    # === 1. 初始化进程组 ===
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    # === 2. 只在主线程进行模型的保存与记录 ===
    if rank == 0:
        ckpt_save_fold = os.path.join(args.project_path, args.ckpt_save_fold)
        epoch_loss_img_save_path = os.path.join(args.project_path, 'all_epoch_loss.png')
        epoch_lr_img_save_path = os.path.join(args.project_path, 'all_epoch_lr.png')
        batch_loss_img_save_path = os.path.join(args.project_path, 'last_epoch_loss.png')
        if not os.path.exists(ckpt_save_fold):
            os.mkdir(ckpt_save_fold)
        best_loss = 1e8
        epoch_loss_list  = []
        epoch_lr_list = []

    # === 3. 准备数据 ===
    total_steps = args.total_steps
    ddpm_dataset_fold = args.ddpm_dataset_fold
    image_transformer = get_image_transformer(args.input_image_size)
    DDPM_Dataset = get_ddpm_dataset(args)
    ddpm_dataset = DDPM_Dataset(ddpm_dataset_fold,
                                 alpha_t_bar_list,
                                 beta_t_bar_list,
                                 total_steps = total_steps,
                                 transform = image_transformer)

    sampler = DistributedSampler(
        ddpm_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    dataloader = torch.utils.data.DataLoader(
        ddpm_dataset,
        batch_size=args.batch_size,
        sampler=sampler,  # 关键：使用分布式采样器
        num_workers=world_size*2,
        pin_memory=True
    )
    total_batchs = len(dataloader)

    # === 3. 创建模型 ===
    net = get_unet(args)
    model = net(args).to(local_rank)
    model = DDP(model, device_ids=[local_rank])  # DDP 包装
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * world_size, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    total_global_steps = total_batchs * args.epochs
    scheduler = CosineAnnealingLR(optimizer, 
                                  T_max= total_global_steps,
                                  eta_min=args.lr * world_size * 0.01)
    if rank == 0:
        logging.info(f"模型参数量为：{count_model_params(model)/1e9} B")
        logging.info(model)
        
    # === 3. 恢复参数及随机状态 ===
    if args.resume_train:
        ckpt_path = args.unet_ckpt_path
        assert os.path.exists(ckpt_path), f"恢复训练指定的ckpt文件：{ckpt_path} 不存在" 
        resume_epoch, best_loss, resume_global_steps = load_checkpoint(ckpt_path, model, optimizer, scheduler, rank)  
        logging.info(f"恢复训练，ckpt路径为: {ckpt_path}")
        logging.info(f"将从第 {resume_epoch+1} epoch, 全局第 {resume_global_steps+1} 步开始训练，当前最佳损失为：{best_loss:.6f}")

    # === 4. 训练循环 ===
    global_steps = 0
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)  # 确保每个 epoch 不同 shuffle
        if rank == 0:
            batch_loss_list  = []
        
        #每个epoch的训练循环
        for batch_index, batch in enumerate(dataloader):
            if args.resume_train and (global_steps <= resume_global_steps):
                global_steps += 1
                continue
            if batch_index % accumulate_grad_batches == 0:
                optimizer.zero_grad()

            #前向传播及损失计算
            images_add_noise, time_step, noises = batch[:3]
            images_add_noise = images_add_noise.to(rank)
            time_step = time_step.to(rank)
            noises = noises.to(rank)

            if len(batch) == 3:
                predict_noise = model(images_add_noise, time_step)  #无文本生成
            elif len(batch) == 4:
                anno_embedding = batch[-1].to(rank)
                predict_noise = model(images_add_noise, time_step, anno_embedding)  #含文本生成
            else:
                raise ValueError(f"希望单个训练数据包含的元素为3或4，得到的元素数量为{len(batch)}，暂不支持")

            local_rank_loss = F.mse_loss(predict_noise, noises, reduction="sum") / len(predict_noise)  #以每张图片的平均损失作为损失值

            local_rank_loss.backward()
            local_loss_tensor = torch.tensor(local_rank_loss.item()).cuda(local_rank)
            dist.all_reduce(local_loss_tensor, op=dist.ReduceOp.SUM)
            all_local_rank_average_loss = float(local_loss_tensor.item() / world_size)  # 全局平均损失
            if rank == 0:    
                logging.info(f"epoch: {epoch}, batch: {batch_index}/{total_batchs}, global steps: {global_steps}/{total_global_steps}, loss: {all_local_rank_average_loss:.6f}, best_loss: {best_loss:.6f}, lr: {float(scheduler.get_last_lr()[0]):.20f}")
                batch_loss_list.append(all_local_rank_average_loss) 
            optimizer.step()
            scheduler.step()
            global_steps += 1

        #每个epoch保存相关数据
        if ((not args.resume_train) and rank == 0) or \
            (args.resume_train and (global_steps-1 > resume_global_steps) and (rank==0)):
            #更新最好模型
            epoch_average_loss = float(np.average(batch_loss_list))
            if best_loss > epoch_average_loss:
                best_loss = epoch_average_loss
                best_epoch_ckpt_save_path = os.path.join(ckpt_save_fold, f"best_ckpt.pth")
                if save_ckpt:
                    save_checkpoint(epoch, batch_index,global_steps-1,best_loss,model,optimizer,scheduler,best_epoch_ckpt_save_path)
            #更新最近一轮模型
            epoch_loss_list.append(epoch_average_loss)
            epoch_lr_list.append(scheduler.get_last_lr())
            least_epoch_ckpt_save_path = os.path.join(ckpt_save_fold, f"least_epoch_ckpt.pth")
            if save_ckpt:
                save_checkpoint(epoch, batch_index,global_steps-1,best_loss,model,optimizer,scheduler,least_epoch_ckpt_save_path)
            #绘制损失和学习率曲线
            plot_training_loss(epoch_loss_img_save_path, epoch_loss_list) #每个epoch的loss
            plot_training_loss(batch_loss_img_save_path, batch_loss_list) #最后一个epoch的每个batch的损失
            plot_training_loss(epoch_lr_img_save_path, epoch_lr_list) #每个epoch的lr

        if ((epoch == args.epochs -1) and (not args.resume_train) and rank == 0) or \
            ((epoch == args.epochs -1) and args.resume_train and (global_steps-1 > resume_global_steps) and (rank==0)):
            last_epoch_ckpt_save_path = os.path.join(ckpt_save_fold, f"last_epoch_ckpt.pth")
            save_checkpoint(epoch, batch_index,global_steps-1,best_loss,model,optimizer,scheduler,last_epoch_ckpt_save_path)
        
        #每n个epoch进行一次推理验证，推理结果图片放在project下
        if ((not args.resume_train) or \
            (args.resume_train and (global_steps-1 > resume_global_steps))) and (epoch % 25 == 0):
            if rank == 0:
                ddpm_validation(model, args, epoch)
            torch.cuda.empty_cache()

    # === 5. 清理 ===
    dist.destroy_process_group()


def count_model_params(model):
    return sum(param.numel() for param in model.parameters())


def computer_vae_kl_loss(mu, log_var, clamp = False):
    if clamp:
        log_var = torch.clamp(log_var, min=-20, max=20)
    return 0.5 * (mu.pow(2) + log_var.exp() -1 - log_var).mean()
    # return 0.5 * (mu.pow(2) + log_var.exp() -1 - log_var).sum(dim=(1,2,3)).mean()



def computer_vae_loss(perception_model, input_image, rebuilt_image, mu, log_var, clamp=False):
    loss_weight = [1, 1, 1e-6]
    rank = int(os.environ["RANK"])
    # rebuilt_loss = torch.abs(input_image - rebuilt_image).mean()
    rebuilt_loss = torch.abs(input_image - rebuilt_image).mean()
    kl_loss = computer_vae_kl_loss(mu, log_var, clamp=clamp)
    perception_loss = perception_model(input_image, rebuilt_image).mean()
    loss = rebuilt_loss * loss_weight[0] +  perception_loss * loss_weight[1] + kl_loss * loss_weight[2]
    if rank == 0:
        logging.info(f"VAE loss weights : {loss_weight}, mu: {mu.detach().cpu().mean().item()}, log_var: {log_var.detach().cpu().mean().item()}")
    return loss, rebuilt_loss, kl_loss, perception_loss


def multi_gpu_vae_train(args, save_ckpt = True, net=VAE):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    accumulate_grad_batches = args.accumulate_grad_batches
    # === 1. 初始化进程组 ===
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    # === 2. 只在主线程进行模型的保存与记录 ===
    if rank == 0:
        ckpt_save_fold = os.path.join(args.project_path, args.ckpt_save_fold)
        epoch_loss_img_save_path = os.path.join(args.project_path, 'all_epoch_loss.png')
        epoch_lr_img_save_path = os.path.join(args.project_path, 'all_epoch_lr.png')
        batch_loss_img_save_path = os.path.join(args.project_path, 'last_epoch_loss.png')
        if not os.path.exists(ckpt_save_fold):
            os.mkdir(ckpt_save_fold)
        best_loss = 1e8
        epoch_loss_list  = []
        epoch_lr_list = []

    # === 3. 准备数据 ===
    train_image_fold = args.train_image_fold
    total_steps = args.total_steps
    image_transformer = get_image_transformer(args.input_image_size)
    celebA_HQ_dataset = VAE_Dataset(train_image_fold, transform=image_transformer)
    sampler = DistributedSampler(
        celebA_HQ_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    dataloader = torch.utils.data.DataLoader(
        celebA_HQ_dataset,
        batch_size=args.batch_size,
        sampler=sampler,  # 关键：使用分布式采样器
        num_workers=world_size*2,
        pin_memory=True
    )
    total_batchs = len(dataloader)

    # === 3. 创建模型 ===
    model = net(args).to(local_rank)
    # perception_model = LPIPS(rank, "yolox")  # 使用自己实现的lpips模型计算损失
    perception_model = LPIPS(rank, "vgg16")  # 使用自己实现的lpips模型计算损失
    # perception_model = lpips.LPIPS(net="vgg").to(rank)  #使用官方的lpips模型计算损失

    model = DDP(model, device_ids=[local_rank])  # DDP 包装
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * world_size, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    total_global_steps = total_batchs * args.epochs
    scheduler = CosineAnnealingLR(optimizer, 
                                  T_max= total_global_steps,
                                  eta_min=args.lr * world_size * 0.01)
    if rank == 0:
        logging.info(f"模型参数量为：{count_model_params(model)/1e9} B")
        logging.info(model)
        
    # === 3. 恢复参数及随机状态 ===
    if args.resume_train:
        ckpt_path = args.vae_ckpt_path
        assert os.path.exists(ckpt_path), f"恢复训练指定的ckpt文件：{ckpt_path} 不存在" 
        resume_epoch, best_loss, resume_global_steps = load_checkpoint(ckpt_path, model, optimizer, scheduler, rank)  
        logging.info(f"恢复训练，ckpt路径为: {ckpt_path}")
        logging.info(f"将从第 {resume_epoch+1} epoch, 全局第 {resume_global_steps+1} 步开始训练，当前最佳损失为：{best_loss:.6f}")

    # === 4. 训练循环 ===
    global_steps = 0
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)  # 确保每个 epoch 不同 shuffle
        if rank == 0:
            batch_loss_list  = []
        
        #每个epoch的训练循环
        for batch_index, batch in enumerate(dataloader):
            if args.resume_train and (global_steps <= resume_global_steps):
                global_steps += 1
                continue
            x = batch
            x = x.cuda(rank, non_blocking=True)  # 关键：使用非阻塞方式，让数据从cpu 到 gpu是异步的
            if batch_index % accumulate_grad_batches == 0:
                optimizer.zero_grad()
            pred_imgs, mu, log_var = model(x)
            #computer vae losss
            loss, rebuilt_loss, kl_loss, perception_loss = computer_vae_loss(perception_model, x, pred_imgs, mu, log_var)
            # lpips_loss = perception_model2(x, pred_imgs).mean()  #与官方LPIPS结果对比，观察自己实现的是否正确
            # print("lpips_loss: ", lpips_loss.item(), " perception_loss: ", perception_loss.item())

            scaler = accumulate_grad_batches if batch_index < (total_batchs//accumulate_grad_batches * accumulate_grad_batches) else total_batchs % accumulate_grad_batches
            loss /= scaler
            loss.backward()
            local_loss_tensor = torch.tensor(loss.item()).cuda(local_rank)
            dist.all_reduce(local_loss_tensor, op=dist.ReduceOp.SUM)
            real_loss = float(local_loss_tensor.item() / world_size * scaler)  # 全局平均损失
            if rank == 0:    
                logging.info(f"epoch: {epoch}, batch: {batch_index}/{total_batchs}, global steps: {global_steps}/{total_global_steps}, loss: {real_loss:.6f}, best_loss: {best_loss:.6f}, rebuilt_loss: {float(rebuilt_loss.item()):.6f}, kl_loss: {float(kl_loss.item()):.6f}, perception_loss: {perception_loss.item():.6f}, lr: {float(scheduler.get_last_lr()[0]):.20f}, accumulate_grad_batches: {scaler}")
                batch_loss_list.append(real_loss) 
            if (batch_index % accumulate_grad_batches == (accumulate_grad_batches - 1)) or (batch_index == total_batchs -1):
                optimizer.step()
            scheduler.step()
            global_steps += 1

        #每个epoch保存相关数据
        if ((not args.resume_train) and rank == 0) or \
            (args.resume_train and (global_steps-1 > resume_global_steps) and (rank==0)):
            #更新最好模型
            epoch_average_loss = float(np.average(batch_loss_list))
            if best_loss > epoch_average_loss:
                best_loss = epoch_average_loss
                best_epoch_ckpt_save_path = os.path.join(ckpt_save_fold, f"best_ckpt.pth")
                if save_ckpt:
                    save_checkpoint(epoch, batch_index,global_steps-1,best_loss,model,optimizer,scheduler,best_epoch_ckpt_save_path)
            #更新最近一轮模型
            epoch_loss_list.append(epoch_average_loss)
            epoch_lr_list.append(scheduler.get_last_lr())
            least_epoch_ckpt_save_path = os.path.join(ckpt_save_fold, f"least_epoch_ckpt.pth")
            if save_ckpt:
                save_checkpoint(epoch, batch_index,global_steps-1,best_loss,model,optimizer,scheduler,least_epoch_ckpt_save_path)
            #绘制损失和学习率曲线
            plot_training_loss(epoch_loss_img_save_path, epoch_loss_list) #每个epoch的loss
            plot_training_loss(batch_loss_img_save_path, batch_loss_list) #最后一个epoch的每个batch的损失
            plot_training_loss(epoch_lr_img_save_path, epoch_lr_list) #每个epoch的lr

        if ((epoch == args.epochs -1) and (not args.resume_train) and rank == 0) or \
            ((epoch == args.epochs -1) and args.resume_train and (global_steps-1 > resume_global_steps) and (rank==0)):
            last_epoch_ckpt_save_path = os.path.join(ckpt_save_fold, f"last_epoch_ckpt.pth")
            save_checkpoint(epoch, batch_index,global_steps-1,best_loss,model,optimizer,scheduler,last_epoch_ckpt_save_path)
    # === 5. 清理 ===
    dist.destroy_process_group()


def check_image_embeddings(args):
    image_save_path = "./infer_img/image_embeddings_check_result.jpg"
    image_embeddings_json = "./dataset/flickr30kr/flickr30k_annotations/image_embeddings_info_512_to_64.json"
    image_npy_path = "./dataset/flickr30kr/flickr30k_annotations/image_embeddings_512_to_64.npy"
    vae_model_path = "./checkpoint/vae_model/best_ckpt.pth"
    with open(image_embeddings_json, 'r', encoding='utf-8') as f:
        json_content = json.load(f)
    try:
        image_embeddings_shape = tuple(json_content["embedding_shape"])
        image_embeddings_dtype = json_content["embedding_dtype"]
    except:
        raise ValueError(f'{image_embeddings_json}文件缺失embedding_shape和embedding_dtype两个字段，无法加载{image_npy_path}文件')
    
    if  image_embeddings_dtype not in ["float32", "float16"]:
        raise ValueError(f'{image_embeddings_json}文件中数据类型与预期不匹配，只支持float32和float16，不支持{image_embeddings_dtype}，无法加载{image_npy_path}文件')

    device = args.device
    image_embeddings_dtype = np.float16 if image_embeddings_dtype == "float16" else np.float32
    image_embeddings = np.memmap(image_npy_path, dtype=image_embeddings_dtype, mode='r', shape=image_embeddings_shape)
    image_index = random.randint(0, len(image_embeddings)-1)
    # image_index = 1
    image_embedding = torch.tensor(image_embeddings[image_index], dtype=torch.float32).unsqueeze(0).to(device)

    # vae_model = VAE(args)
    # vae_encoder_model = ldm_load_model(vae_model_path, vae_model).encoder.to(device)
    # vae_encoder_model.eval()
    # print("image_embedding: ", image_embedding)
    # image_path = "./dataset/flickr30kr/flickr30k_images_512/images/10002456.jpg"
    # image_trans = get_image_transformer(512)
    # input_image_embedding = image_trans(Image.open(image_path)).unsqueeze(0).to(device)
    # with torch.no_grad():
    #     input_image_embedding = vae_encoder_model(input_image_embedding)
    # mu = input_image_embedding[:, :4]
    # print("mu: ", mu)

    vae_model = VAE(args)
    vae_decoder_model = ldm_load_model(vae_model_path, vae_model).decoder.to(device)
    vae_decoder_model.eval()
    with torch.no_grad():
        pred_x = vae_decoder_model(image_embedding)

    #保存最终图片
    image = detransform_tensor2image(pred_x.detach().cpu())
    concatenated_image = concatenate_images_horizontally(image)
    concatenated_image.save(image_save_path)



def vae_inference(args, net=VAE):
    #支持同时推理多张图片
    device = get_device(args.device)
    model_path = args.vae_ckpt_path
    vae_infer_image_path = args.vae_infer_image_path
    image_transformers = get_image_transformer(args.input_image_size)
    if os.path.isfile(vae_infer_image_path):
        x = Image.open(vae_infer_image_path)
        x_in = image_transformers(x).unsqueeze(0).to(device)
    else:
        raise ValueError(f"输入的图片路径不是一个文件：{vae_infer_image_path}")

    model = net(args)
    model = ldm_load_model(model_path, model).to(device)
    model.eval()
    
    infer_times = int(args.infer_times)
    batch_size = int(args.batch_size)
    inference_image_save_fold = args.inference_image_save_fold
    with torch.no_grad():
        print("VAE开始推理...")
        image_name = f"vae_infer.jpg"
        image_save_path = os.path.join(inference_image_save_fold, image_name)
        x_pred = model(x_in)[0]
        
        #保存最终图片
        x_list = torch.cat([x_in, x_pred], dim = 0)
        image_list = detransform_tensor2image(x_list.detach().cpu())
        concatenated_image = concatenate_images_horizontally(image_list)
        concatenated_image.save(image_save_path)

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--batch_size', type=int, default=4)
    argparse.add_argument('--epochs', type=int, default=2500)
    argparse.add_argument('--lr', type=float, default=1e-4)
    argparse.add_argument('--time_embedding_dims', type=int, default=128)
    argparse.add_argument('--input_image_size', type=int, default=512)
    argparse.add_argument('--input_image_dims', type=int, default=3)
    argparse.add_argument('--output_image_dims', type=int, default=3)
    argparse.add_argument('--time_steps', type=int, default=1000)
    argparse.add_argument('--save_every_epochs', type=int, default=50)
    argparse.add_argument('--ckpt_save_fold', type=str, default='checkpoints')
    argparse.add_argument('--base_fold', type=str, default='./project')
    argparse.add_argument("--device", type=str, default="cuda:3")
    argparse.add_argument("--train_image_fold", type=str, default="./dataset/flickr30kr/flickr30k_images_512/images")
    argparse.add_argument("--ddpm_dataset_fold", type=str, default="./dataset/flickr30kr/flickr30k_images_128")
    argparse.add_argument("--total_steps", type=int, default=1000)
    argparse.add_argument("--mode", type=str, default="multi_gpu_vae_train")
    argparse.add_argument("--vae_ckpt_path", type=str, default="")
    argparse.add_argument("--unet_ckpt_path", type=str, default="")
    argparse.add_argument("--infer_times", type=int, default=10)
    argparse.add_argument("--denoise_steps_gap", type=int, default=50)
    argparse.add_argument("--inference_image_save_fold", type=str, default="./infer_img")
    argparse.add_argument("--train_mse_loss_mode", type=str, default="mean", help="train mse loss mode mean or sum")
    argparse.add_argument("--inference_show_denoise_image_every_n_steps", type=int, default=20)
    argparse.add_argument("--train_activate_func", type=str, default="silu")
    argparse.add_argument("--resume_train", action='store_true', help="whether to resume training")
    argparse.add_argument("--accumulate_grad_batches", type=int, default=8)
    argparse.add_argument("--vae_infer_image_path", type=str, default="")
    argparse.add_argument("--embedding_model", type=str, default="clip")
    argparse.add_argument("--ddpm_prompt_list", nargs='+', help="list of ddpm prompt")
    args = argparse.parse_args()

    if args.mode == "ddpm_train" or args.mode == "multi_gpu_ddpm_train" or args.mode == "multi_gpu_vae_train":
        rank = os.environ.get("RANK", None)
        save_ckpt = True
        if rank is None or int(rank) == 0:
            #启动日志
            current_time = datetime.datetime.now()
            time_str = current_time.strftime('%Y%m%d_%H%M%S_%f')
            project_path = os.path.join(args.base_fold, time_str)
            args.project_path = project_path
            if not os.path.exists(project_path):
                os.mkdir(project_path)
            log_path = os.path.join(project_path, 'log.txt')
            logging.basicConfig(filename = log_path ,level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            #记录参数
            args_dict = vars(args)
            logging.info(f"rank: {rank}")
            args_str = '\n'.join(f'{key}: {value}' for key, value in args_dict.items())
            logging.info('Command line arguments:')
            logging.info(args_str)
        #训练
        if args.mode == "ddpm_train":
            ddpm1_train(args)  
        elif args.mode == "multi_gpu_ddpm_train":
            multi_gpu_ddpm_train(args, save_ckpt=save_ckpt)
        elif args.mode == "multi_gpu_vae_train":
            multi_gpu_vae_train(args, save_ckpt=save_ckpt)
    elif args.mode == "ddpm_infer":
        net = get_unet(args)
        model_path = args.unet_ckpt_path
        model = net(args)
        model = ldm_load_model(model_path, model)
        device = get_device(args.device)
        model = model.to(device)
        ddpm_inference(model, args)  #ddpm推理
    elif args.mode == "ldm_infer":
        ldm_inference(args)  #ddpm推理
    elif args.mode == "vae_infer":
        vae_inference(args)  #vae推理
    elif args.mode == "show_add_noise_process":
        show_noise_image()  #可视化加噪过程
    elif args.mode == "valid_denoise_process":
        show_demoise_image()  #可视化去噪过程，验证去噪公式是否有效
    elif args.mode == "test":
        test()
    elif args.mode == "computrt_fid":
        test_computer_fid_loss(args)
    elif args.mode == "check_vae_encoder_result":
        check_image_embeddings(args)
        