import torch
from glob import glob
import os
import logging
from tqdm import tqdm
import json
import numpy as np
import random 
from PIL import Image
from utils.utils import clip_vit_base_patch32_model_infer
from utils.ddpm_schedule import add_noise


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
        assert len(self.train_images) == 1, f'{self.dataset_fold}下有多个json文件或缺失json文件'
        self.train_images = self.train_images[0]  #1、标注embedding数据的形状、数据类型信息，2、图像名称和索引对于关系

        self.anno_csv_path = "./dataset/flickr30kr/flickr30k_annotations/text_annotations.csv"  #标注信息原始数据
        self.anno_npz_path = "./dataset/flickr30kr/flickr30k_annotations/text_embeddings.npz"  #标注信息转换成的embedding数据，np.memmap格式
        
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
        # for img_name in tqdm(img_name_list):
        #     img_path = os.path.join(self.image_dir, img_name)
        #     if not os.path.isfile(img_path):
        #         self.image_paths = []
        #         raise ValueError(f"图片{img_path}不存在，请检查")
        #     else:
        #         self.image_paths.append(img_path)
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
        logging.info(f"开始使用{input_words_list}获取空文本embeddings")
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
    
    

class LDM_Flickr30K_CLIP_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_fold, shift_func, noise_func, total_steps = 1000, transform = None):
        super().__init__()
        self.dataset_fold = dataset_fold
        self.embeddings_drop_ratio = 0.1
        
        #图像相关
        self.transform = transform
        self.total_steps = total_steps
        self.shift_func = shift_func
        self.noise_func = noise_func

        #标注相关
        json_file_paths = glob(os.path.join(self.dataset_fold, '*.json'))
        self.image_embeddings_json = None
        self.text_embeddings_json = None
        self.image_list_json = None
        for json_file_path in json_file_paths:
            json_file_name = json_file_path.split(os.path.sep)[-1]
            if json_file_name.startswith("image_embeddings"):
                self.image_embeddings_json = json_file_path
            elif json_file_name.startswith("text_embeddings"):
                self.text_embeddings_json = json_file_path
            elif json_file_name.startswith("image_list"):
                self.image_list_json = json_file_path
            else:
                raise ValueError(f"只支持以 image_embeddings | text_embeddings | image_list 开头的json文件")
        assert len(json_file_paths) == 3, f'{self.dataset_fold}下有{len(json_file_paths)}个json文件，数量不对，应该为3个'
        if (self.image_embeddings_json is None) or  (self.text_embeddings_json is None) or (self.image_list_json is None):
            raise ValueError(f"部分json文件缺失。image_embeddings：{self.image_embeddings_json} | text_embeddings：{self.text_embeddings_json} | image_list：{self.image_list_json}")
        
        self.anno_csv_path = os.path.join(self.dataset_fold, "text_annotations.csv")  #标注信息原始数据
        self.anno_npy_path = os.path.join(self.dataset_fold, "text_embeddings.npy")  #标注信息转换成的embedding数据，np.memmap格式
        self.image_npy_path = os.path.join(self.dataset_fold, "image_embeddings_512_to_64.npy")
        logging.info(f"-----当前数据集为：{self.__class__}-----")
        logging.info(f"标注原始数据为{self.anno_csv_path}")
        logging.info(f"标注embeddings数据路径为{self.anno_npy_path}")
        logging.info(f"标注embeddings数据信息json文件路径为{self.text_embeddings_json}")
        logging.info(f"图像embeddings数据路径为{self.image_npy_path}")
        logging.info(f"图像embeddings数据信息json文件路径为{self.image_embeddings_json}")
        logging.info(f"数据集使用的图像列表路径为{self.image_list_json}")

        logging.info("-----开始加载并检查数据-----")
        self.load_text_embeddings()
        self.load_image_embeddings()
        self.check_text_embeddings()
        self.check_image_embeddings()
        self.get_image_embedding_mean_std()
        self.get_empty_text_embeddings()
        logging.info("-----数据加载完毕，所有检查通过-----")


    def load_text_embeddings(self):
        logging.info("-----开始加载text embeddings数据-----")
        with open(self.text_embeddings_json, 'r', encoding='utf-8') as f:
            json_content = json.load(f)
        try:
            self.text_embeddings_shape = tuple(json_content["embedding_shape"])
            self.text_embeddings_dtype = json_content["embedding_dtype"]
        except:
            raise ValueError(f'{self.text_embeddings_json}文件缺失embedding_shape和embedding_dtype两个字段，无法加载{self.anno_npy_path}文件')
        
        self.every_image_n_annotations, self.max_token_num, self.embeddings_dim = [int(i) for i in self.text_embeddings_shape[1:]]
        if  self.text_embeddings_dtype not in ["float32", "float16"]:
            raise ValueError(f'{self.text_embeddings_json}文件中数据类型与预期不匹配，只支持float32和float16，不支持{self.text_embeddings_dtype}，无法加载{self.anno_npz_path}文件')
        
        self.text_embeddings_dtype = np.float16 if self.text_embeddings_dtype == "float16" else np.float32
        self.text_embeddings = np.memmap(self.anno_npy_path, dtype=self.text_embeddings_dtype, mode='r', shape=self.text_embeddings_shape)
        logging.info("-----加载text embeddings数据完毕-----")


    def load_image_embeddings(self):
        logging.info("-----开始加载image embeddings数据-----")
        with open(self.image_embeddings_json, 'r', encoding='utf-8') as f:
            json_content = json.load(f)
        try:
            self.image_embeddings_shape = tuple(json_content["embedding_shape"])
            self.image_embeddings_dtype = json_content["embedding_dtype"]
        except:
            raise ValueError(f'{self.image_embeddings_json}文件缺失embedding_shape和embedding_dtype两个字段，无法加载{self.image_npy_path}文件')
        
        if  self.image_embeddings_dtype not in ["float32", "float16"]:
            raise ValueError(f'{self.image_embeddings_json}文件中数据类型与预期不匹配，只支持float32和float16，不支持{self.image_embeddings_dtype}，无法加载{self.image_npz_path}文件')
        
        self.image_embeddings_dtype = np.float16 if self.image_embeddings_dtype == "float16" else np.float32
        self.image_embeddings = np.memmap(self.image_npy_path, dtype=self.image_embeddings_dtype, mode='r', shape=self.image_embeddings_shape)
        logging.info("-----加载image embeddings数据完毕-----")

    
    def get_empty_text_embeddings(self):
        input_words_list = [""]
        logging.info(f"-----开始使用{input_words_list}获取空文本embeddings-----")
        clip_result = clip_vit_base_patch32_model_infer(input_words_list, device='cpu')
        clip_result = clip_result.squeeze().numpy()
        if clip_result.shape != (self.max_token_num, self.embeddings_dim):
            raise ValueError(f"获取空文本embeddings失败，返回的embedding维度为{clip_result.shape}，不是{self.max_token_num, self.embeddings_dim}")
        else:
            logging.info(f"获取空文本embeddings成功，返回的空文本embedding的值为{clip_result}")
        self.empty_text_embeddings = clip_result
        return self.empty_text_embeddings

    def random_check_text_embeddings(self, input_words_list, input_embedding):
        clip_result = clip_vit_base_patch32_model_infer(input_words_list, device='cpu')
        clip_result = clip_result.squeeze().numpy().astype(self.text_embeddings_dtype)
        input_embedding = input_embedding.squeeze()
        logging.info(f"随机选取的标注内容：{input_words_list}")
        logging.info(f"实际结果：{clip_result}")
        logging.info(f"读取内容：{input_embedding}")
        if np.any(abs(clip_result-input_embedding) > 1e-2):
            return False
        else:
            return True

    def check_text_embeddings(self):
        #随机选取一个标注，检查其内容是否和clip的原始输出内容一致
        logging.info("-----开始检查随机选取的一个text的text embedding是否和经过clip输出的embedding一致-----")
        with open(self.image_list_json, 'r', encoding='utf-8') as f:
            image_name_dict = json.load(f)
        random_image_index = random.randint(0, self.text_embeddings_shape[0])
        random_anno_index = random.randint(0,self.every_image_n_annotations-1)
        random_image_name = image_name_dict[str(random_image_index)]
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

        input_embedding = self.text_embeddings[random_image_index][random_anno_index]
        input_words_list = [annotation]

        if self.random_check_text_embeddings(input_words_list, input_embedding):
            logging.info(f"随机选取的图片为：{random_image_name}，随机选取的标注为:{annotation}，和clip的原始输出内容一致")
        else:
            raise ValueError("数据检查失败")        

    def check_image_embeddings(self):
        assert len(self.image_embeddings) == len(self.text_embeddings), f"图像和标注的数量不一致，图片的数量为{len(self.image_embeddings)}，标注的数量为{len(self.text_embeddings)}"
        self.image_nums = len(self.image_embeddings)

    def get_image_embedding_mean_std(self):
        # 计算图片经过vae的encoder之后的隐空间表示的均值和方差，用于缩放image embedding的方差到1，以保证加噪过程有效
        logging.info("-----开始计算图片经过vae的encoder之后的隐空间表示的均值和方差-----")
        self.image_mean = np.mean(self.image_embeddings)
        self.image_std = np.std(self.image_embeddings.astype(np.float64))
        logging.info(f"经过vae encoder之后的图片embedding的均值为{self.image_mean}，方差为{self.image_std}")
        return self.image_mean, self.image_std


    def __len__(self):
        return self.image_nums
    
    def add_noise(self, x, t, verbose=False):
        return add_noise(x, t, self.shift_func, self.noise_func, verbose=verbose)

    def __getitem__(self, index):
        #图片经过vae的encoder之后的输出不在[-1,1]之间，这里使用整个图片数据集的标准差进行缩放，然后再对图片进行加噪 
        #缩放后整个数据集中图片embedding的标准差接近1，加上vae训练得到的分布理论上均值接近0，经过转换后整个图片的分布接近均值0方差1
        #均值0方差1的分布约有66%的数值在【-1,1】之间，通过这种方式来匹配原有加噪强度
        sacaled_image_embedding = torch.tensor(self.image_embeddings[index] / self.image_std, dtype=torch.float32)        
        t = torch.randint(0, self.total_steps, (1,))  #torch.randint不包含最后一个数
        image_embedding, _, noise = self.add_noise(sacaled_image_embedding, t)
        if random.random() > self.embeddings_drop_ratio:
            text_embedding = self.text_embeddings[index][random.randint(0, self.every_image_n_annotations-1)]
        else:
            text_embedding = self.empty_text_embeddings
        text_embedding = torch.tensor(text_embedding, dtype=torch.float32)
        return image_embedding, t, noise, text_embedding
    