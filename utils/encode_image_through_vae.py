import os
import numpy as np
import argparse
import sys
sys.path.append("/data1/zhenyu/hand_write_aigc")
from model.vae import KL_VAE as VAE
from utils import vae_model_args_name_type_dict
from utils import save_embedding_to_memmap
from utils import get_image_transformer
from utils import ldm_load_model
import json
from PIL import Image
from tqdm import tqdm
import torch

def get_model_args(log_file_path, args_name_dict):
    args_dict = {}
    with open(log_file_path, 'r', encoding='utf-8') as f:
        max_line_num = 0
        while (max_line_num < 1000):
            line = f.readline()
            if line and line.startswith(tuple(args_name_dict.keys())):
                line_content_list = line.split(":")
                assert len(line_content_list) == 2, f"模型参数解析错误，解析错误内容为{line}"
                key, value = line_content_list
                key = key.strip()
                value = value.strip()
                if key in args_name_dict:
                    if args_name_dict[key].lower() != "str":
                        value = eval(args_name_dict[key] + "(" + value.strip() + ")")
                    else:
                        value = value.strip()
                    args_dict[key] = value
            if len(args_dict) == len(args_name_dict):
                break
            else:
                max_line_num += 1
    found_args = set(args_dict.keys())
    needed_args = set(args_name_dict.keys())
    if found_args != needed_args:
        raise ValueError(f"模型参数缺失，需要的参数有{needed_args}，但是只找到了{found_args}")
    model_args = argparse.Namespace(**args_dict)
    return model_args

def get_image_embeddings(vae_encoder, image_list_json_file, img_source_fold, image_transformers, device):
    img_embeddings_list = []
    with open(image_list_json_file, "r", encoding="utf-8") as f:
        iamge_index_name_dict = json.load(f)
    if iamge_index_name_dict is None:
        raise ValueError(f"{image_list_json_file}是空文件")
    for i in tqdm(range(len(iamge_index_name_dict))):
        key = str(i)
        if key not in iamge_index_name_dict:
            raise ValueError(f"图片索引名称字典中没有找到索引：{i}")
        with torch.no_grad():
            img_path = os.path.join(img_source_fold, iamge_index_name_dict[str(i)])
            img = image_transformers(Image.open(img_path))
            img = img.unsqueeze(0).to(device)
            vae_encoder_out = vae_encoder(img)
            mu, log_var = vae_encoder_out[:, :4], vae_encoder_out[:, 4:]
            mu = mu.detach().squeeze().cpu().numpy()
            img_embeddings_list.append(mu)
    img_embeddings = np.stack(img_embeddings_list, axis=0)
    return img_embeddings


if __name__ == "__main__":
    project_root = "./checkpoint/vae_model"
    ckpt_path = os.path.join(project_root, "best_ckpt.pth")
    log_file_path = os.path.join(project_root, "log.txt")
    model_args_name_type_dict = vae_model_args_name_type_dict
    device = "cuda:3"

    img_source_fold = "./dataset/flickr30kr/flickr30k_images_512/images"
    image_list_json_file = "./dataset/flickr30kr/flickr30k_annotations/image_list.json"
    img_embeddings_save_path = "./dataset/flickr30kr/flickr30k_annotations/image_embeddings_512_to_64.npy"
    img_embeddings_info_save_path = "./dataset/flickr30kr/flickr30k_annotations/image_embeddings_info_512_to_64.json"

    vae_model_args = get_model_args(log_file_path, model_args_name_type_dict)
    assert hasattr(vae_model_args, "input_image_size"), f"{vae_model_args}中没有找到input_image_size参数，无法加载图片预处理模块"
    image_transformers = get_image_transformer(vae_model_args.input_image_size)


    vae_model = VAE(vae_model_args)
    vae_encoder = ldm_load_model(ckpt_path, vae_model).encoder.to(device).eval()

    image_embeddings = get_image_embeddings(vae_encoder, image_list_json_file, img_source_fold, image_transformers, device)
    image_embeddings_info = save_embedding_to_memmap(image_embeddings, img_embeddings_save_path, use_float16=False)
    with open(img_embeddings_info_save_path, "w", encoding="utf-8") as f:
        json.dump(image_embeddings_info, f, indent=4)
    print("Done!")
    print(f"图片embeddings已存储至{img_embeddings_save_path}")
    print(f"图片embeddings信息已保存在{img_embeddings_info_save_path}")




            
        

