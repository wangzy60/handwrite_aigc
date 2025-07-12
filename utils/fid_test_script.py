import torch
import torch.nn as nn
from torchvision.models import inception_v3
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import numpy as np
from scipy.linalg import sqrtm
import os
from glob import glob
from utils.utils import get_device


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
    conv_real = np.cov(real_features, rowvar=True)  #计算协方差矩阵
    conv_fake = np.cov(fake_features, rowvar=True)  #计算协方差矩阵
    conv_mean = sqrtm(conv_real @ conv_fake)  # 计算矩阵平方根。矩阵平方根的意思是1个方形矩阵和它自身的矩阵乘积，是另外1个矩阵。
    if not np.isfinite(conv_mean).all():
        #如果矩阵的平方根求出来有些值溢出了，则添加一个很小的数，然后再计算矩阵平方根
        epsilon = np.eye(conv_real.shape[0]) * 1e-6
        conv_mean = sqrtm((conv_real + epsilon) @ (conv_fake + epsilon))
    conv_mean = conv_mean.real
    # np.trace(a + b - c) = np.trace(a) + np.trace(b) - np.trace(c)
    #trace是矩阵的迹，即对角线元素之和
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
    
    