from PIL import Image
import os
from glob import glob
from tqdm import tqdm
from torchvision import transforms


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


if __name__ == "__main__":
    img_size = 128
    src_dir = "./dataset/flickr30kr/flickr30k_images"
    dst_dir = f"./dataset/flickr30kr/flickr30k_images_{img_size}/images"
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    image_path_list = glob(os.path.join(src_dir, "*.jpg"))
    for image_path in tqdm(image_path_list):
        save_image_path = os.path.join(dst_dir, image_path.split("/")[-1])
        img_transformers = ResizeAutoPad(img_size, (128, 128, 128))
        img = Image.open(image_path)
        padding_img = img_transformers(img)
        padding_img.save(save_image_path)
    
    