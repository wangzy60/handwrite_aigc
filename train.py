import torch.distributed
from utils.utils import get_device
from utils.ddpm_schedule import alpha_t_list,alpha_t_bar_list,beta_t_list,beta_t_bar_list
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
# from model.unet import Unet_With_Text_Condition as Unet
from model.unet import Unet_Without_Condition as Unet
# from model.unet import Unet2 as Unet

from model.vae import KL_VAE as VAE


import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision.models import inception_v3
from torchvision.models import vgg16
import lpips
from checkpoint.lpips_weights import lpips_offical_weights
from scipy.linalg import sqrtm

#绘制最近一个epoch的损失曲线:done
#学习率调度:done
#训练恢复:done
#恢复训练后减少恢复步数需要的时间
#FID损失计算
#DDIM采样实现
#IS损失计算        

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
    
    

def add_noise(x, t, shift_func, noise_func, verbose=False):
    if isinstance(shift_func, np.ndarray) and isinstance(noise_func, np.ndarray):
        shift_value = shift_func[t]
        noise_value = noise_func[t]
    else:
        shift_value = shift_func(t)
        noise_value = noise_func(t)
        
    noise = torch.randn_like(x)
    noise_img = noise_value*noise
    img_with_noisy = shift_value*x + noise_img
    return img_with_noisy, noise_img, noise


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


def clip_vit_base_patch32_model_infer(input_words_list, device='cpu'):
    from transformers import AutoModel, AutoTokenizer
    model_name = "./checkpoint/clip-vit-base-patch32"
    text_model = AutoModel.from_pretrained(model_name).text_model.to(device)
    tokenize = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenize(input_words_list, padding="max_length", max_length=77, return_tensors = "pt").to(device)
    with torch.no_grad():
        embeddings = text_model(**tokens)
    clip_result = embeddings.last_hidden_state.detach()
    return clip_result


class DDPM_Flickr30K_CLIP_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_fold, shift_func, noise_func, total_steps = 1000, transform = None):
        super().__init__()
        self.dataset_fold = dataset_fold
        self.every_image_n_annotations = 5
        self.max_token_num = 77
        self.embeddings_dim = 512
        self.embeddings_drop_ratio = 0.1

        #标注相关
        self.train_images = glob(os.path.join(self.dataset_fold, '*.json'))
        assert len(self.train_images) == 1, f'{self.annotation_dir}下有多个json文件或缺失json文件'
        self.train_images = self.train_images[0]  #1、标注embedding数据的形状、数据类型信息，2、图像名称和索引对于关系

        self.anno_csv_path = "./dataset/flickr30kr/flickr30k_annotations/all_raw_annotations.csv"  #标注信息原始数据
        self.anno_npz_path = "./dataset/flickr30kr/flickr30k_annotations/all_embedding_annotations.npz"  #标注信息转换成的embedding数据，np.memmap格式
        


        #图像相关
        self.image_dir = os.path.join(dataset_fold, "images")
        self.image_paths = []
        self.transform = transform
        self.total_steps = total_steps
        self.shift_func = shift_func
        self.noise_func = noise_func

        logging.info("-----开始加载并检查标注embeddings数据-----")
        self.load_annotation()
        self.load_image_paths()
        self.check_annotation()
        self.empty_text_embeddings = self.get_empty_text_embeddings()
        logging.info("-----数据加载完毕，所有检查通过-----")


    def load_annotation(self):
        with open(self.train_images, 'r', encoding='utf-8') as f:
            json_content = json.load(f)
            if "memmap_info" in json_content and "image_index_to_name" in json_content:
                embedding_info = json_content["memmap_info"]
                image_index_to_name = json_content["image_index_to_name"]
            else:
                raise ValueError(f'{self.train_images}文件中memmap_info字段或image_index_to_name字段缺失，无法加载{self.anno_npz_path}文件')
        
        if embedding_info and ("embedding_shape" in embedding_info) and ("embedding_dtype" in embedding_info):
            self.embeddings_shape = tuple(embedding_info["embedding_shape"])
            self.embeddings_dtype = embedding_info["embedding_dtype"]
        else:
            raise ValueError(f'{self.train_images}文件中memmap_info字段缺失embedding_shape和embedding_dtype两个字段，无法加载{self.anno_npz_path}文件')
        
        if self.embeddings_shape[1:] != (self.every_image_n_annotations, self.max_token_num, self.embeddings_dim) or self.embeddings_dtype not in ["float32", "float16"]:
            raise ValueError('f{self.train_images}文件中memmap_info字段下的形状或数据类型与预期不匹配，无法加载{self.anno_npz_path}文件')
        
        self.embeddings_dtype = np.float16 if self.embeddings_dtype == "float16" else np.float32
        self.embeddings = np.memmap(self.anno_npz_path, dtype=self.embeddings_dtype, mode='r', shape=self.embeddings_shape)

        self.image_index_to_name = image_index_to_name
        logging.info("标注embeddings数据加载完成")


    def load_image_paths(self):
        logging.info("开始检查标注文件中图片是否都存在")
        img_name_list = list(self.image_index_to_name.values())
        for img_name in tqdm(img_name_list):
            img_path = os.path.join(self.image_dir, img_name)
            if not os.path.isfile(img_path):
                self.image_paths = []
                raise ValueError(f"图片{img_path}不存在，请检查")
            else:
                self.image_paths.append(img_path)
        logging.info("图片名字和索引对照表检查通过")

        logging.info(f'数据集加载了目录{self.dataset_fold}下的{len(self.image_paths)}张图像')
        logging.info(f'前20张图像地址为：{self.image_paths[:20] if len(self.image_paths) > 20 else self.image_paths}')


    def random_check_embeddings(self, input_words_list, input_embedding):
        clip_result = clip_vit_base_patch32_model_infer(input_words_list, device='cpu')
        clip_result = clip_result.squeeze().numpy().astype(self.embeddings_dtype)
        input_embedding = input_embedding.squeeze()
        logging.info(f"随机选取的标注内容：{input_words_list}")
        logging.info(f"实际结果：{clip_result}")
        logging.info(f"读取内容：{input_embedding}")
        if np.any(abs(clip_result-input_embedding) > 1e-2):
            return False
        else:
            return True
    
    def get_empty_text_embeddings(self):
        input_words_list = [""]
        logging.info("f开始使用{input_words_list}获取空文本embeddings")
        clip_result = clip_vit_base_patch32_model_infer(input_words_list, device='cpu')
        clip_result = clip_result.squeeze().numpy()
        if clip_result.shape != (self.max_token_num, self.embeddings_dim):
            raise ValueError(f"获取空文本embeddings失败，返回的embedding维度为{clip_result.shape}，不是{self.max_token_num, self.embeddings_dim}")
        else:
            logging.info(f"获取空文本embeddings成功，返回的空文本embedding的值为{clip_result}")
        return clip_result


    def check_annotation(self):
        #随机选取一个标注，检查其内容是否和clip的原始输出内容一致
        logging.info("开始检查随机选取的一个标注是否和clip的原始输出内容一致")
        random_image_index = random.randint(0, len(self.image_index_to_name.keys())-1)
        random_anno_index = random.randint(0,self.every_image_n_annotations-1)
        random_image_name = self.image_index_to_name[str(random_image_index)]
        annotation = None
        with open(self.anno_csv_path, 'r', encoding='utf-8') as f:
            csv_content = f.readlines()
            for i in range(len(csv_content)):
                if csv_content[i].startswith(random_image_name):
                    annotation = csv_content[i + random_anno_index]
                    annotation = annotation.split("|")[-1].replace("\n", " ").strip()
                    break
        if annotation is None:
            raise ValueError("没有找到随机的图片及其对应的标注, 请检查代码")

        input_embedding = self.embeddings[random_image_index][random_anno_index]
        input_words_list = [annotation]

        if self.random_check_embeddings(input_words_list, input_embedding):
            logging.info(f"随机选取的图片为：{random_image_name}，随机选取的标注为:{annotation}，和clip的原始输出内容一致")
        else:
            raise ValueError("数据检查失败")        

    def __len__(self):
        return len(self.image_paths)
    
    def add_noise(self, x, t, verbose=False):
        return add_noise(x, t, self.shift_func, self.noise_func, verbose=verbose)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        t = torch.randint(0, self.total_steps, (1,))  #torch.randint不包含最后一个数
        img_with_noisy, _, noise = self.add_noise(image, t)
        if random.random() > self.embeddings_drop_ratio:
            picture_annotation_embedding = self.embeddings[index][random.randint(0, self.every_image_n_annotations-1)]
        else:
            picture_annotation_embedding = self.empty_text_embeddings
        picture_annotation_embedding = torch.tensor(picture_annotation_embedding, dtype=torch.float)
        return img_with_noisy, t, noise, picture_annotation_embedding
    

class DDPM_Flickr30K_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_fold, shift_func, noise_func, total_steps = 1000, transform = None):
        super().__init__()
        self.dataset_fold = dataset_fold
        #图像相关
        self.image_paths = glob(os.path.join(self.dataset_fold, "*.jpg"))
        self.transform = transform
        self.total_steps = total_steps
        self.shift_func = shift_func
        self.noise_func = noise_func

        logging.info(f'数据集加载了目录{self.dataset_fold}下的{len(self.image_paths)}张图像')
        logging.info(f'前20张图像地址为：{self.image_paths[:20] if len(self.image_paths) > 20 else self.image_paths}')

    def __len__(self):
        return len(self.image_paths)
    
    def add_noise(self, x, t, verbose=False):
        return add_noise(x, t, self.shift_func, self.noise_func, verbose=verbose)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        t = torch.randint(0, self.total_steps, (1,))  #torch.randint不包含最后一个数
        img_with_noisy, _, noise = self.add_noise(image, t)
        return img_with_noisy, t, noise


class DDPM_CelebAHQ_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_fold, shift_func, noise_func, total_steps = 1000, transform = None):
        super().__init__()
        #图像相关
        self.image_dir = dataset_fold
        self.image_paths = glob(os.path.join(self.image_dir, '*.jpg'))
        self.transform = transform
        self.total_steps = total_steps
        self.shift_func = shift_func
        self.noise_func = noise_func

        logging.info(f'数据集加载了目录{self.dataset_fold}下的{len(self.image_paths)}张图像')
        logging.info(f'前20张图像地址为：{self.image_paths[:20] if len(self.image_paths) > 20 else self.image_paths}')

        
    def __len__(self):
        return len(self.image_paths)
    
    def add_noise(self, x, t, verbose=False):
        return add_noise(x, t, self.shift_func, self.noise_func, verbose=verbose)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        t = torch.randint(0, self.total_steps, (1,))  #这里需要用total_steps+1，因为torch.randint不包含最后一个数
        img_with_noisy, _, noise = self.add_noise(image, t)
        return img_with_noisy, t, noise



class VAE_Dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform = None):
        super().__init__()
        self.image_paths = (glob(os.path.join(image_dir, '*.jpg')) + glob(os.path.join(image_dir, '*.jpeg')))
        logging.info(f'VAE数据集加载了{len(self.image_paths)}张图像')
        logging.info(f'前20张图像地址为：\n{self.image_paths[:20] if len(self.image_paths) > 20 else self.image_paths}')
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image


class ResizeAutoPad:
    def __init__(self, target_size=512, fill=(128, 128, 128)):
        self.target_size = target_size
        self.fill = fill

    def __call__(self, img):
        img_w, img_h = img.size
        if (img_h == self.target_size) and (img_w == self.target_size):
            return img
        scalar = self.target_size / max(img_h, img_w)
        target_h, target_w = int(img_h * scalar), int(img_w * scalar)
        img = transforms.functional.resize(img, (target_h, target_w), interpolation=Image.LANCZOS)
        img_w, img_h = img.size
        padded_img = Image.new("RGB", (self.target_size, self.target_size), self.fill)
        paste_x = (self.target_size - img_w) // 2
        paste_y = (self.target_size - img_h) // 2        
        padded_img.paste(img, (paste_x, paste_y))
        return padded_img


def get_image_transformer(image_size):
    transform = transforms.Compose([
        ResizeAutoPad(target_size= image_size, fill=(128, 128, 128)),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: (x - 0.5) * 4)
        transforms.Lambda(lambda x: x * 2 - 1)  #一般将图片的值域范围处理到[-1,1]
    ])
    return transform

def detransform_tensor2image(image_tensor):
    #image_tensor(c,h,w) / (b, c, h, w)

    # image_numpy = ((image_tensor / 4.0 + 0.5) * 255.0).to(torch.uint8).numpy()
    image_tensor = torch.clamp(image_tensor, min=-1, max=1)  #这里需要做显示的值域控制，因为torch的.to(torch.uint8)操作不会对溢出的部分进行处理，比如256使用to(torch.uint8)会得到0而不是255
    image_numpy = ((image_tensor + 1) * 255 / 2).to(torch.uint8).numpy()  
    if len(image_tensor.shape) == 3:
        image = Image.fromarray(image_numpy.transpose(1,2,0))
        return image
    elif len(image_tensor.shape) == 4:
        image_numpy = image_numpy.transpose(0,2,3,1)
        image_list = [Image.fromarray(img.squeeze()) for img in image_numpy]
        return image_list
    else:
        raise ValueError(f"去噪图片的tensor形状必须是3维或者4维，否则无法转换，当前维度为{len(image_tensor.shape)}")


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


def ddpm1_train(args, net=Unet):
    ckpt_save_fold = os.path.join(args.project_path, args.ckpt_save_fold)
    loss_img_save_path = os.path.join(args.project_path, 'loss_img.png')
    train_mse_loss_mode = args.train_mse_loss_mode
    if not os.path.exists(ckpt_save_fold):
        os.mkdir(ckpt_save_fold)
    device = get_device(args.device)
    train_image_fold = args.train_image_fold
    total_steps = args.total_steps
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
            

def ddpm_load_model(model_path, net):
    saved_params = torch.load(model_path, weights_only=False)
    print(saved_params)
    model_params = saved_params.get('model_state_dict', None)
    if model_params is None:
        raise ValueError("No state_dict found in the checkpoint file.")
    
    # 兼容多卡训练的模型。多卡训练的模型外面会多包裹一层'module.'， 去除key中的'module.'
    new_model_params = {k.replace('module.', ''): v for k, v in model_params.items()}

    model = net(args)
    incompatiable = model.load_state_dict(new_model_params)  #返回的是不匹配的key，不是加载参数后的模型
    if len(incompatiable.missing_keys) == 0 and len(incompatiable.unexpected_keys) == 0:
        print(model)
    elif incompatiable.missing_keys:
        raise ValueError(f"缺失的key：{incompatiable.missing_keys}")
    elif incompatiable.unexpected_keys:
        raise ValueError(f"多余的key：{incompatiable.unexpected_keys}")       
    return model


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



def ddpm1_inference(args, net=Unet):
    #支持同时推理多张图片
    device = get_device(args.device)
    total_steps = int(args.total_steps)
    model_path = args.ckpt_path

    model = ddpm_load_model(model_path, net).to(device)
    model.eval()
    
    infer_times = int(args.infer_times)
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
        
        #保存过程图片
        if (inference_show_denoise_image_every_n_steps is not None):
            image_list.append(image)
            concatenated_image = [concatenate_images_vertical(image) for image in image_list]
            concatenated_image = concatenate_images_horizontally(concatenated_image)
            concat_image_save_path = os.path.join(inference_image_save_fold, f"infer_gap_{denoise_steps_gap}_every_{inference_show_denoise_image_every_n_steps}_steps.png")
            concatenated_image.save(concat_image_save_path)


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
        checkpoint_dict = torch.load(ckpt_path)
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
        

def multi_gpu_ddpm_train(args, save_ckpt = True, net=Unet):
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
    ddpm_dataset_fold = args.ddpm_dataset_fold
    image_transformer = get_image_transformer(args.input_image_size)
    # ddpm_dataset = DDPM_Flickr30K_CLIP_Dataset(ddpm_dataset_fold, 
    #                                         alpha_t_bar_list, 
    #                                         beta_t_bar_list, 
    #                                         total_steps=total_steps, 
    #                                         transform=image_transformer)
    ddpm_dataset = DDPM_Flickr30K_Dataset(ddpm_dataset_fold, 
                                        alpha_t_bar_list, 
                                        beta_t_bar_list, 
                                        total_steps=total_steps, 
                                        transform=image_transformer)
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
        ckpt_path = args.ckpt_path
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

            local_rank_loss = F.mse_loss(predict_noise, noises, reduction="mean")

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
    # === 5. 清理 ===
    dist.destroy_process_group()


def count_model_params(model):
    return sum(param.numel() for param in model.parameters())


def count_params(args):
    model = Unet(args)
    print(f"模型参数量为: {count_model_params(model)/1e9} B")
    # from torchinfo import summary
    # summary(model) 


def load_vgg16_model(rank, n_layers=16):
    if rank == 0:
        vgg_model = vgg16(pretrained=True).features[:n_layers]
        state_dict = vgg_model.state_dict()
        object_list = [state_dict]
    else:
        vgg_model = vgg16(pretrained=False).features[:n_layers]
        state_dict = None
        object_list = [state_dict]

    torch.distributed.broadcast_object_list(object_list, src=0)
    torch.distributed.barrier()

    if rank != 0:
        state_dict = object_list[0]
        vgg_model.load_state_dict(state_dict)

    vgg_model.eval()
    for param in vgg_model.parameters():
        param.requires_grad_(False)
    vgg_model = vgg_model.to(rank)
    return vgg_model


def computer_vae_kl_loss(mu, log_var, clamp = False):
    if clamp:
        log_var = torch.clamp(log_var, min=-20, max=20)
    return 0.5 * (mu.pow(2) + log_var.exp() -1 - log_var).mean()
    # return 0.5 * (mu.pow(2) + log_var.exp() -1 - log_var).sum(dim=(1,2,3)).mean()


class LPIPS(nn.Module):
    def __init__(self, rank, model_name, lpips_weights = lpips_offical_weights, yolox_weight_path = "./checkpoint/yolox_backbone.pth"):
        super().__init__()
        self.model_name = model_name
        if model_name == "vgg16":
            self.n_layers = 30
            self.rank = rank
            self.channel_weights_list = lpips_weights
            if rank == 0:
                vgg_model = vgg16(pretrained=True).features[:self.n_layers]
                state_dict = vgg_model.state_dict()
                object_list = [state_dict]
            else:
                vgg_model = vgg16(pretrained=False).features[:self.n_layers]
                state_dict = None
                object_list = [state_dict]

            torch.distributed.broadcast_object_list(object_list, src=0)
            torch.distributed.barrier()

            if rank != 0:
                state_dict = object_list[0]
                vgg_model.load_state_dict(state_dict)

            vgg_model.eval()
            for param in vgg_model.parameters():
                param.requires_grad_(False)
            
            children_model_list = list(vgg_model.children())
            self.conv1 = nn.Sequential(*children_model_list[:4]).to(rank)
            self.conv2 = nn.Sequential(*children_model_list[4:9]).to(rank)
            self.conv3 = nn.Sequential(*children_model_list[9:16]).to(rank)
            self.conv4 = nn.Sequential(*children_model_list[16:23]).to(rank)
            self.conv5 = nn.Sequential(*children_model_list[23:30]).to(rank)
            
            assert len(self.channel_weights_list) == 5, "channel_weights_list的长度必须为5，以匹配5个特征层"
            self.L = len(self.channel_weights_list)
            
            for i in range(len(self.channel_weights_list)):
                self.channel_weights_list[i].requires_grad = False
                self.channel_weights_list[i] = self.channel_weights_list[i].to(rank)
        elif model_name == "yolox":
            from model.yolox import CSPDarknet
            self.yolox_model = CSPDarknet()
            if rank == 0:
                yolox_backbone_statedict = self.abstract_yolox_backbone_statedict(yolox_weight_path)
                yolox_backbone_statedict_list = [yolox_backbone_statedict]
            else:
                yolox_backbone_statedict = None
                yolox_backbone_statedict_list = [yolox_backbone_statedict]
            torch.distributed.broadcast_object_list(yolox_backbone_statedict_list, src=0)
            torch.distributed.barrier()
            self.yolox_model.load_state_dict(yolox_backbone_statedict_list[0], strict=True)
            self.yolox_model.eval()
            for param in self.yolox_model.parameters():
                param.requires_grad_(False)
            self.yolox_model = self.yolox_model.to(rank)
        else:
            raise ValueError("model_name must be vgg16 or yolox, but got {}".format(model_name))

    def abstract_yolox_backbone_statedict(self, pre_trained_model_path):
        pre_trained_state_dict = torch.load(pre_trained_model_path)
        pre_str = "backbone.backbone."
        new_state_dict = {k.split(pre_str)[1]:v for k, v in pre_trained_state_dict['model'].items() if k.startswith(pre_str)}
        return new_state_dict
    
    def vae_input_image_to_yolox_input_image(self, vae_input_image):
        device = vae_input_image.device
        yolox_input_image = ((vae_input_image + 1) * 0.5 * 255).to(device)
        return yolox_input_image

    def vae_input_image_to_vgg_input_image(self, vae_input_image):
        device = vae_input_image.device
        vgg_input_image = (vae_input_image + 1) * 0.5
        vgg_input_image = (vgg_input_image - torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device))/torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)
        return vgg_input_image

    def safe_L2_norm(self, input_tensor, epsilon = 1e-10):
        scalar = torch.sqrt(torch.sum(input_tensor ** 2, dim=1, keepdim = True) + epsilon)
        return input_tensor/scalar

    def forward(self, input_image, pred_image):
        if self.model_name == "vgg16":
            #将图像像素值从[-1,1]转换成满足均值0方差1的正态分布，满足vgg的图像输入要求
            vgg_input_image = self.vae_input_image_to_vgg_input_image(input_image)
            vgg_pred_image = self.vae_input_image_to_vgg_input_image(pred_image)
            #获取原图和重建图的多尺度特征图
            input_feature1 = self.conv1(vgg_input_image)
            input_feature2 = self.conv2(input_feature1)
            input_feature3 = self.conv3(input_feature2)
            input_feature4 = self.conv4(input_feature3)
            input_feature5 = self.conv5(input_feature4)
            input_image_feature_list = [input_feature1, input_feature2, input_feature3, input_feature4, input_feature5]
            pred_feature1 = self.conv1(vgg_pred_image)
            pred_feature2 = self.conv2(pred_feature1)
            pred_feature3 = self.conv3(pred_feature2)
            pred_feature4 = self.conv4(pred_feature3)
            pred_feature5 = self.conv5(pred_feature4)
            pred_image_feature_list = [pred_feature1, pred_feature2, pred_feature3, pred_feature4, pred_feature5]
            #对所有特征在通道维度进行L2归一化
            input_image_feature_l2_norm_list, pred_image_feature_l2_norm_list = [], []
            for input_feature, pred_feature in zip(input_image_feature_list, pred_image_feature_list):
                input_image_feature_l2_norm_list.append(self.safe_L2_norm(input_feature))
                pred_image_feature_l2_norm_list.append(self.safe_L2_norm(pred_feature))
            #计算输入和预测特征图的L2差别
            diff_list = []
            for input_feature, pred_feature in zip(input_image_feature_l2_norm_list, pred_image_feature_l2_norm_list):
                diff = input_feature - pred_feature
                diff_list.append(diff**2)
            #根据不同层的权重，计算最终的感知损失
            loss = 0.0
            for i in range(self.L):
                loss += torch.sum(self.channel_weights_list[i].view(1,-1,1,1) * diff_list[i], dim = 1, keepdim=True).mean(dim=(1,2,3))
        elif self.model_name == "yolox":
            yolox_input_image = self.vae_input_image_to_yolox_input_image(input_image)
            yolox_pred_image = self.vae_input_image_to_yolox_input_image(pred_image)
            input_image_feature_list = list(self.yolox_model(yolox_input_image).values())
            pred_image_feature_list = list(self.yolox_model(yolox_pred_image).values())
            #对所有特征在通道维度进行L2归一化
            input_image_feature_l2_norm_list, pred_image_feature_l2_norm_list = [], []
            for input_feature, pred_feature in zip(input_image_feature_list, pred_image_feature_list):
                input_image_feature_l2_norm_list.append(self.safe_L2_norm(input_feature))
                pred_image_feature_l2_norm_list.append(self.safe_L2_norm(pred_feature))
            #计算输入和预测特征图的L2差别
            diff_list = []
            for input_feature, pred_feature in zip(input_image_feature_l2_norm_list, pred_image_feature_l2_norm_list):
                diff = input_feature - pred_feature
                diff_list.append(diff**2)
            #这里直接使用不同通道的特征差异作为权重损失，去除尺寸对特征的影响，保留通道数量对特征的影响
            # loss = torch.sum(torch.cat([diff.mean(dim=(2,3)) for diff in diff_list], dim=1), dim=1, keepdim=True)
            loss = torch.sum(torch.cat([diff.mean(dim=(1,2,3), keepdim=True) for diff in diff_list], dim=1), dim=1, keepdim=True)
        else:
            raise ValueError("model_name must be vgg16 or yolox, but got {}".format(self.model_name))
        return loss.mean()


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
        ckpt_path = args.ckpt_path
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


def vae_inference(args, net=VAE):
    #支持同时推理多张图片
    device = get_device(args.device)
    model_path = args.ckpt_path
    vae_infer_image_path = args.vae_infer_image_path
    image_transformers = get_image_transformer(args.input_image_size)
    if os.path.isfile(vae_infer_image_path):
        x = Image.open(vae_infer_image_path)
        x_in = image_transformers(x).unsqueeze(0).to(device)
    else:
        raise ValueError(f"输入的图片路径不是一个文件：{vae_infer_image_path}")

    model = ddpm_load_model(model_path, net).to(device)
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
    argparse.add_argument("--ddpm_dataset_fold", type=str, default="./dataset/flickr30kr/flickr30k_images_512_test")
    argparse.add_argument("--total_steps", type=int, default=1000)
    argparse.add_argument("--mode", type=str, default="multi_gpu_vae_train")
    argparse.add_argument("--ckpt_path", type=str, default="/shared_file/hand_write_aigc/project/20250711_231553_966794/checkpoints/least_epoch_ckpt.pth")
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
        ddpm1_inference(args)  #ddpm推理
    elif args.mode == "vae_infer":
        vae_inference(args)  #vae推理
    elif args.mode == "show_add_noise_process":
        show_noise_image()  #可视化加噪过程
    elif args.mode == "valid_denoise_process":
        show_demoise_image()  #可视化去噪过程，验证去噪公式是否有效
    elif args.mode == "test":
        test()
    elif args.mode == "count_params":
        count_params(args)
    elif args.mode == "computrt_fid":
        test_computer_fid_loss(args)

