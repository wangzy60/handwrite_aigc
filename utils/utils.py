import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import gc

vae_model_args_name_type_dict = {"input_image_size":"int", 
                                "input_image_dims":"int", 
                                "train_activate_func":"str"}

def get_device(device_param:str):
    device = torch.device(device_param) if torch.cuda.is_available() and device_param.startswith("cuda") else torch.device("cpu")
    return device


def save_embedding_to_memmap(embedding_array, save_path, use_float16 = True):
    #输入embeddings数组，保存路径，将embeddings保存到memmap中，返回embeddings的shape和dtype信息
    max_num = np.max(embedding_array)
    min_num = np.min(embedding_array)
    print(f"embedding最大值为{max_num},最小值为{min_num}")
    if use_float16 and (max_num < 65504) and (min_num > -65504):
        print(f"符合float16值域要求，将embedding转换为float16类型存储，以节约空间")
        embedding_dtype = np.float16
    else:
        print(f"不符合float16值域要求，embedding将以float32类型存储")
        embedding_dtype = np.float32
    memmap = np.memmap(save_path, dtype=embedding_dtype, mode='w+', shape=embedding_array.shape)
    chunk_size = embedding_array.shape[0] if embedding_array.shape[0] < 5000 else 5000
    for i in range(0, embedding_array.shape[0], chunk_size):
        memmap[i:i + chunk_size] = embedding_array[i:i + chunk_size]
    memmap.flush()
    del memmap  # 关闭内存映射
    print(f'embeddings保存成功，保存路径为：{save_path}')

    memmap_info = {
        "embedding_shape": list(embedding_array.shape),
        "embedding_dtype": str(embedding_dtype.__name__)
    }
    return memmap_info


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


def clip_vit_base_patch32_model_infer(input_words_list, device='cpu'):
    from transformers import AutoModel, AutoTokenizer
    model_name = "./checkpoint/clip-vit-base-patch32"
    text_model = AutoModel.from_pretrained(model_name).text_model.to(device)
    tokenize = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenize(input_words_list, padding="max_length", max_length=77, return_tensors = "pt").to(device)
    with torch.no_grad():
        embeddings = text_model(**tokens)
    clip_result = embeddings.last_hidden_state.detach()
    # 卸载模型
    del text_model
    del tokenize
    gc.collect()  # 强制垃圾回收
    return clip_result


def ldm_load_model(model_path, model):
    saved_params = torch.load(model_path, weights_only=False)
    print(saved_params)
    model_params = saved_params.get('model_state_dict', None)
    if model_params is None:
        raise ValueError("No state_dict found in the checkpoint file.")
    
    # 兼容多卡训练的模型。多卡训练的模型外面会多包裹一层'module.'， 去除key中的'module.'
    new_model_params = {k.replace('module.', ''): v for k, v in model_params.items()}

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
    if isinstance(image_numpy_list, list):
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
    elif image_numpy_list:
        return image_numpy_list
    else:
        raise ValueError("输入不能为空")