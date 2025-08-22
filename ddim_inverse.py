from model.unet import Unet2 as Unet
import numpy as np
import argparse
import torch
from utils.ddpm_schedule import add_noise
from utils.utils import clip_vit_base_patch32_model_infer, get_device, get_image_transformer, detransform_tensor2image, ldm_load_model, concatenate_images_horizontally, concatenate_images_vertical
from utils.ddpm_schedule import alpha_t_list,alpha_t_bar_list,beta_t_list,beta_t_bar_list
from tqdm import tqdm
import os
from PIL import Image


def ddpm_inverse(image_path, model, args):
    #处理图片
    img = Image.open(image_path)
    transformer = get_image_transformer(img.width)
    x_t = transformer(img)
    
    #ddim inverse
    device = next(model.parameters()).device
    total_steps = int(args.total_steps)
    model.eval()
    batch_size = int(args.batch_size)
    ddpm_prompt_list = args.ddpm_prompt_list
    denoise_steps_gap = int(args.denoise_steps_gap)
    inference_image_save_fold = args.inference_image_save_fold
    inference_show_denoise_image_every_n_steps = args.inference_show_denoise_image_every_n_steps
    with torch.no_grad():
        print("开始ddim inverse...")
        image_list = []
        image_save_taggle = 0
        if ddpm_prompt_list:
            embeddings = clip_vit_base_patch32_model_infer(ddpm_prompt_list, device)
        image_name = f"ddim_inverse_gap_{denoise_steps_gap}.jpg"
        image_save_path = os.path.join(inference_image_save_fold, image_name)
        time_schedule = list(range(1, total_steps, denoise_steps_gap))
        if time_schedule[-1] != len(alpha_t_bar_list) -1:
            time_schedule.append(len(alpha_t_bar_list) -1)
        image_list.append(detransform_tensor2image(x_t.detach().cpu()))
        print("time_schedule: ", time_schedule)
        ##ddim
        for i in tqdm(range(len(time_schedule))):
            t = time_schedule[i]
            time_step = torch.tensor([t]*batch_size).to(device)
            # predict_noise = model(x_t, time_step, embeddings)
            predict_noise = model(x_t, time_step)
            if i+1 < len(time_schedule):
                n_t = time_schedule[i+1]
            else:
                break
            print(f"t: {t}, next_t: {n_t}")
            a_t_bar = float(alpha_t_bar_list[t])
            a_nt_bar = float(alpha_t_bar_list[n_t])
            b_t_bar = float(beta_t_bar_list[t])
            b_nt_bar = float(beta_t_bar_list[n_t])
            x_t = (a_nt_bar * x_t + (a_t_bar*b_nt_bar - a_nt_bar * b_t_bar) * predict_noise) / (a_t_bar)

            #保存过程图片
            if (inference_show_denoise_image_every_n_steps is not None) and (t > image_save_taggle):
                image_save_taggle += inference_show_denoise_image_every_n_steps
                image_list.append(detransform_tensor2image(x_t.detach().cpu()))        

        #保存最终图片
        image = detransform_tensor2image(x_t.detach().cpu())
        concatenated_image = concatenate_images_horizontally(image)
        concatenated_image.save(image_save_path)
        
        #保存过程图片
        if (inference_show_denoise_image_every_n_steps is not None):
            image_list.append(image)
            concatenated_image = [concatenate_images_vertical(image) for image in image_list]
            concatenated_image = concatenate_images_horizontally(concatenated_image)
            concat_image_save_path = os.path.join(inference_image_save_fold, f"ddim_reverse_gap_{denoise_steps_gap}_every_{inference_show_denoise_image_every_n_steps}_steps.png")
            concatenated_image.save(concat_image_save_path)
    return x_t, model, time_schedule


def ddim(x_t, model, time_schedule, args):
    #处理图片
    device = next(model.parameters()).device
    total_steps = int(args.total_steps)
    model.eval()
    batch_size = int(args.batch_size)
    ddpm_prompt_list = args.ddpm_prompt_list
    denoise_steps_gap = int(args.denoise_steps_gap)
    inference_image_save_fold = args.inference_image_save_fold
    inference_show_denoise_image_every_n_steps = args.inference_show_denoise_image_every_n_steps
    with torch.no_grad():
        print("开始ddim 推理...")
        image_list = []
        image_save_taggle = total_steps
        if ddpm_prompt_list:
            embeddings = clip_vit_base_patch32_model_infer(ddpm_prompt_list, device)
        image_name = f"ddim_inverse_inverse_gap_{denoise_steps_gap}.jpg"
        image_save_path = os.path.join(inference_image_save_fold, image_name)
        image_list.append(detransform_tensor2image(x_t.detach().cpu()))
        time_schedule = time_schedule[::-1]
        print("time_schedule: ", time_schedule)
        ##ddim
        for i in tqdm(range(len(time_schedule))):
            t = time_schedule[i]
            time_step = torch.tensor([t]*batch_size).to(device)
            # predict_noise = model(x_t, time_step, embeddings)
            predict_noise = model(x_t, time_step)
            if i+1 < len(time_schedule):
                p_t = time_schedule[i+1]
            else:
                break
            print(f"t: {t}, next_t: {p_t}")
            a_t_bar = float(alpha_t_bar_list[t])
            a_pt_bar = float(alpha_t_bar_list[p_t])
            b_t_bar = float(beta_t_bar_list[t])
            b_pt_bar = float(beta_t_bar_list[p_t])
            x_t = a_pt_bar / a_t_bar * (x_t - b_t_bar * predict_noise) + b_pt_bar * predict_noise

            #保存过程图片
            if (inference_show_denoise_image_every_n_steps is not None) and (t < image_save_taggle):
                image_save_taggle -= inference_show_denoise_image_every_n_steps
                image_list.append(detransform_tensor2image(x_t.detach().cpu()))        

        #保存最终图片
        image = detransform_tensor2image(x_t.detach().cpu())
        concatenated_image = concatenate_images_horizontally(image)
        concatenated_image.save(image_save_path)
        
        #保存过程图片
        if (inference_show_denoise_image_every_n_steps is not None):
            image_list.append(image)
            concatenated_image = [concatenate_images_vertical(image) for image in image_list]
            concatenated_image = concatenate_images_horizontally(concatenated_image)
            concat_image_save_path = os.path.join(inference_image_save_fold, f"ddim_reverse_inverse_gap_{denoise_steps_gap}_every_{inference_show_denoise_image_every_n_steps}_steps.png")
            concatenated_image.save(concat_image_save_path)
    return x_t



def ddpm_inverse_test(image_path, model, args):
    #处理图片
    img = Image.open(image_path)
    transformer = get_image_transformer(img.width)
    x_t = transformer(img)
    
    #ddim inverse
    device = next(model.parameters()).device
    total_steps = int(args.total_steps)
    model.eval()
    batch_size = int(args.batch_size)
    ddpm_prompt_list = args.ddpm_prompt_list
    denoise_steps_gap = int(args.denoise_steps_gap)
    inference_image_save_fold = args.inference_image_save_fold
    inference_show_denoise_image_every_n_steps = args.inference_show_denoise_image_every_n_steps
    with torch.no_grad():
        print("开始ddim inverse...")
        image_list = []
        image_save_taggle = 0
        if ddpm_prompt_list:
            embeddings = clip_vit_base_patch32_model_infer(ddpm_prompt_list, device)
        image_name = f"ddim_inverse_gap_{denoise_steps_gap}.jpg"
        image_save_path = os.path.join(inference_image_save_fold, image_name)
        time_schedule = list(range(1, total_steps, 1))
        if time_schedule[-1] != len(alpha_t_bar_list) -1:
            time_schedule.append(len(alpha_t_bar_list) -1)
        image_list.append(detransform_tensor2image(x_t.detach().cpu()))
        print("time_schedule: ", time_schedule)
        ##ddim
        for i in tqdm(range(len(time_schedule))):
            t = time_schedule[i]
            time_step = torch.tensor([t]*batch_size).to(device)
            # predict_noise = model(x_t, time_step, embeddings)
            predict_noise = model(x_t, time_step)
            if i+1 < len(time_schedule):
                n_t = time_schedule[i+1]
            else:
                break
            print(f"t: {t}, next_t: {n_t}")
            a_t = float(alpha_t_list[t])
            b_t = float(beta_t_list[t])
            x_t = a_t * x_t + b_t * predict_noise

            #保存过程图片
            if (inference_show_denoise_image_every_n_steps is not None) and (t > image_save_taggle):
                image_save_taggle += inference_show_denoise_image_every_n_steps
                image_list.append(detransform_tensor2image(x_t.detach().cpu()))        

        #保存最终图片
        image = detransform_tensor2image(x_t.detach().cpu())
        concatenated_image = concatenate_images_horizontally(image)
        concatenated_image.save(image_save_path)
        
        #保存过程图片
        if (inference_show_denoise_image_every_n_steps is not None):
            image_list.append(image)
            concatenated_image = [concatenate_images_vertical(image) for image in image_list]
            concatenated_image = concatenate_images_horizontally(concatenated_image)
            concat_image_save_path = os.path.join(inference_image_save_fold, f"ddim_reverse_gap_{denoise_steps_gap}_every_{inference_show_denoise_image_every_n_steps}_steps.png")
            concatenated_image.save(concat_image_save_path)
    return x_t, model, time_schedule


def ddim_test(x_t, model, time_schedule, args):
    #处理图片
    device = next(model.parameters()).device
    total_steps = int(args.total_steps)
    model.eval()
    batch_size = int(args.batch_size)
    ddpm_prompt_list = args.ddpm_prompt_list
    denoise_steps_gap = int(args.denoise_steps_gap)
    inference_image_save_fold = args.inference_image_save_fold
    inference_show_denoise_image_every_n_steps = args.inference_show_denoise_image_every_n_steps
    with torch.no_grad():
        print("开始ddim 推理...")
        image_list = []
        image_save_taggle = total_steps
        if ddpm_prompt_list:
            embeddings = clip_vit_base_patch32_model_infer(ddpm_prompt_list, device)
        image_name = f"ddim_inverse_inverse_gap_{denoise_steps_gap}.jpg"
        image_save_path = os.path.join(inference_image_save_fold, image_name)
        image_list.append(detransform_tensor2image(x_t.detach().cpu()))
        time_schedule = list(range(1, total_steps, denoise_steps_gap))
        if time_schedule[-1] != len(alpha_t_bar_list) -1:
            time_schedule.append(len(alpha_t_bar_list) -1)
        time_schedule = time_schedule[::-1]
        print("time_schedule: ", time_schedule)
        ##ddim
        for i in tqdm(range(len(time_schedule))):
            t = time_schedule[i]
            time_step = torch.tensor([t]*batch_size).to(device)
            # predict_noise = model(x_t, time_step, embeddings)
            predict_noise = model(x_t, time_step)
            if i+1 < len(time_schedule):
                p_t = time_schedule[i+1]
            else:
                break
            print(f"t: {t}, next_t: {p_t}")
            a_t_bar = float(alpha_t_bar_list[t])
            a_pt_bar = float(alpha_t_bar_list[p_t])
            b_t_bar = float(beta_t_bar_list[t])
            b_pt_bar = float(beta_t_bar_list[p_t])
            x_t = a_pt_bar / a_t_bar * (x_t - b_t_bar * predict_noise) + b_pt_bar * predict_noise

            #保存过程图片
            if (inference_show_denoise_image_every_n_steps is not None) and (t < image_save_taggle):
                image_save_taggle -= inference_show_denoise_image_every_n_steps
                image_list.append(detransform_tensor2image(x_t.detach().cpu()))        

        #保存最终图片
        image = detransform_tensor2image(x_t.detach().cpu())
        concatenated_image = concatenate_images_horizontally(image)
        concatenated_image.save(image_save_path)
        
        #保存过程图片
        if (inference_show_denoise_image_every_n_steps is not None):
            image_list.append(image)
            concatenated_image = [concatenate_images_vertical(image) for image in image_list]
            concatenated_image = concatenate_images_horizontally(concatenated_image)
            concat_image_save_path = os.path.join(inference_image_save_fold, f"ddim_reverse_inverse_gap_{denoise_steps_gap}_every_{inference_show_denoise_image_every_n_steps}_steps.png")
            concatenated_image.save(concat_image_save_path)
    return x_t




if __name__ == "__main__":
    net = Unet
    model_path = "./checkpoint/img_size_128_epoch_1000_batch_256_ddpm.pth"
    argparse = argparse.ArgumentParser()
    args = argparse.parse_args()
    args.input_image_size = 128
    args.input_image_dims = 3
    args.output_image_dims = 3
    args.time_embedding_dims = 128
    args.train_activate_func = "silu"
    args.time_steps = 1000
    
    args.total_steps = 1000
    args.batch_size = 1
    args.ddpm_prompt_list = " "
    args.denoise_steps_gap = 50
    args.inference_image_save_fold = "./infer_img"
    args.inference_show_denoise_image_every_n_steps = 50
    
    model = net(args)
    model = ldm_load_model(model_path, model)
    image_path = "./dataset/celebA_HQ/data128x128/00002.jpg"
    x_t, model, time_schedule = ddpm_inverse_test(image_path, model, args) 
    ddim_test(x_t, model, time_schedule, args) 
    