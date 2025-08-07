### ddpm训练

torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=12345 train.py --mode multi_gpu_ddpm_train --input_image_size 128 --ddpm_dataset_fold ./dataset/flickr30kr/flickr30k_images_128/images --epochs 3000 --batch_size 45 --input_image_dims 3 --output_image_dims 3

### ddpm恢复训练

torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=12345 train.py --mode multi_gpu_ddpm_train --resume_train --unet_ckpt_path ./project/20250803_090253_637954/checkpoints/least_epoch_ckpt.pth --input_image_size 128 --ddpm_dataset_fold ./dataset/flickr30kr/flickr30k_images_128/images --epochs 3000 --batch_size 45 --input_image_dims 3 --output_image_dims 3

### ddpm 推理

python train.py --mode ddpm_infer --unet_ckpt_path ./project/20250726_140216_538920/checkpoints/best_ckpt.pth  --batch_size 10 --input_image_size 128 --input_image_dims 3 --output_image_dims 3 --denoise_steps_gap 1

### ldm训练

torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=12345 train.py --mode multi_gpu_ddpm_train --input_image_size 64 --ddpm_dataset_fold ./dataset/flickr30kr/flickr30k_annotations --epochs 3000 --batch_size 136 --input_image_dims 4 --output_image_dims 4

### ldm 推理

python train.py --mode ldm_infer --unet_ckpt_path ./project/20250720_211033_232901/checkpoints/best_ckpt.pth --vae_ckpt_path ./checkpoint/vae_model/best_ckpt.pth --batch_size 4 --input_image_size 64 --input_image_dims 4 --output_image_dims 4 --denoise_steps_gap 1 --ddpm_prompt_list "Two young guys with shaggy hair look at their hands while hanging out in the yard ." "Several men in hard hats are operating a giant pulley system ." "A child in a pink dress is climbing up a set of stairs in an entry way ." "Someone in a blue shirt and hat is standing on stair and leaning against a window ."

### VAE 推理

python train.py --mode vae_infer --vae_ckpt_path ./checkpoint/vae_model/best_ckpt.pth --vae_infer_image_path ./dataset/flickr30kr/flickr30k_images_512/images/256063.jpg
