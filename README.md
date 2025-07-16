### 单机多卡训练

torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=12345 train.py

### ddpm训练

torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=12345 train.py --mode multi_gpu_ddpm_train --input_image_size 128 --ddpm_dataset_fold /data1/zhenyu/hand_write_aigc/dataset/flickr30kr/flickr30k_images_128 --epochs 1000 --batch_size 45

### ddpm 推理

python train.py --mode ddpm_infer --ckpt_path /data1/zhenyu/hand_write_aigc/project/20250715_123748_271982/checkpoints/last_epoch_ckpt.pth --batch_size 4 --ddpm_prompt_list "Two young guys with shaggy hair look at their hands while hanging out in the yard ." "Several men in hard hats are operating a giant pulley system ." "A child in a pink dress is climbing up a set of stairs in an entry way ." "Someone in a blue shirt and hat is standing on stair and leaning against a window ." --input_image_size 128 --denoise_steps_gap 1python train.py --mode ddpm_infer --ckpt_path /data1/zhenyu/hand_write_aigc/project/20250715_123748_271982/checkpoints/last_epoch_ckpt.pth --batch_size 4 --ddpm_prompt_list "Two young guys with shaggy hair look at their hands while hanging out in the yard ." "Several men in hard hats are operating a giant pulley system ." "A child in a pink dress is climbing up a set of stairs in an entry way ." "Someone in a blue shirt and hat is standing on stair and leaning against a window ."
