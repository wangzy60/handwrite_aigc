import torch
from transformers import AutoModel, AutoTokenizer
import os
from tqdm import tqdm
from glob import glob
import numpy as np
import json

def get_image_name_and_annotations_dict(annotation_file_path):
    file_name_2_annotations_dict = {}
    with open(annotation_file_path, 'r') as f:
        lines = f.readlines()
    print("处理标注文本……")
    for line in tqdm(lines):
        line_content_list = line.split("|")
        image_name, image_annotation = line_content_list[0].strip(), line_content_list[-1].replace("\n", " ").strip()
        if image_name == "image_name":
            continue
        if image_name in file_name_2_annotations_dict:
            file_name_2_annotations_dict[image_name].append(image_annotation)
        else:
            file_name_2_annotations_dict[image_name] = [image_annotation]
    return file_name_2_annotations_dict


def text_to_embeddings(text_List, tokenizer, text_encode_model, device):
    tokens = tokenizer(text_List, padding="max_length", truncation=True ,max_length=77, return_tensors = "pt").to(device)
    with torch.no_grad():
        embeddings = text_encode_model(**tokens).last_hidden_state.detach().cpu().numpy()
    assert embeddings.shape[0] == len(text_List) and embeddings.shape[1] == 77 and embeddings.shape[2] == 512, "文本转换为embeddings向量过程出错"
    return embeddings


def transform_annotation_to_embedding(file_name_2_annotations_dict, model_path = "clip", device = "cpu"):
    index_to_name_dict = {}
    embeddings_list = []
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    text_encode_model = AutoModel.from_pretrained(model_path).text_model.to(device)
    print("开始将标注转化为embeddings...")
    for index, (image_name, image_annotations) in tqdm(enumerate(file_name_2_annotations_dict.items()), total=len(file_name_2_annotations_dict.items())):
        index_to_name_dict[index] = image_name
        embeddings = text_to_embeddings(image_annotations, tokenizer, text_encode_model, device)
        embeddings_list.append(embeddings)
    embeddings_array = np.stack(embeddings_list, axis = 0)
    return embeddings_array, index_to_name_dict

def save_embedding_to_memmap(embedding_array, save_path):
    max_num = np.max(embedding_array)
    min_num = np.min(embedding_array)
    print(f"embedding最大值为{max_num},最小值为{min_num}")
    if (max_num < 65504) and (min_num > -65504):
        print(f"符合float16值域要求，将embedding转换为float16类型存储，以节约空间")
        embedding_dtype = np.float16
    else:
        print(f"不符合float16值域要求，embedding将以float32类型存储")
        embedding_dtype = np.float32
    print("使用np.memmap存储，避免使用时占用过多内存")
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

def save_index_and_name_dict_to_json_file(index_to_name_dict, memmap_info, save_path):
    save_dict_content = {}
    with open(save_path, 'w', encoding='utf-8') as f:
        save_dict_content["memmap_info"] = memmap_info
        save_dict_content["image_index_to_name"] = index_to_name_dict
        json.dump(save_dict_content, f)
        print(f'memmap信息保存成功，图片索引名称对应信息保存成功，保存路径为：{save_path}')


if __name__ == "__main__":
    device = "cuda:3"
    annotation_file_path = "./dataset/flickr30kr/results.csv"
    embeddings_save_path = "./dataset/flickr30kr/results"
    index_and_name_dict_save_path = "./dataset/flickr30kr/results.json"
    clip_model_dir = "./checkpoint/clip-vit-base-patch32"
    file_name_2_annotations_dict = get_image_name_and_annotations_dict(annotation_file_path)
    embeddings_array, index_to_name_dict = transform_annotation_to_embedding(file_name_2_annotations_dict, model_path = clip_model_dir, device = device)
    memmap_info = save_embedding_to_memmap(embeddings_array, embeddings_save_path+'.npz')
    save_index_and_name_dict_to_json_file(index_to_name_dict, memmap_info, index_and_name_dict_save_path)

    # annotation_file_path = "/shared_file/hand_write_aigc/dataset/flickr30kr/results.csv"
    # inputs = ["a photo of a dog"]
    # model_name = "./checkpoint/clip-vit-base-patch32"
    # device = "cpu"
    # text_model = AutoModel.from_pretrained(model_name).text_model.to(device)
    # tokenize = AutoTokenizer.from_pretrained(model_name)
    # tokens = tokenize(inputs, padding="max_length", max_length=77, return_tensors = "pt").to(device)
    # print(tokens)
    # with torch.no_grad():
    #     embeddings = text_model(**tokens)
    # print(embeddings.last_hidden_state.shape)
    # print(embeddings.last_hidden_state)
    # print(embeddings.pooler_output.shape)