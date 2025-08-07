from model.unet import Unet3
import argparse
import torch
import torch.nn.functional as F

def count_model_params(model):
    return sum(param.numel() for param in model.parameters())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.input_image_size = 128
    args.input_image_dims = 3
    args.output_image_dims = 3
    args.time_embedding_dims = 128
    args.train_activate_func = "silu"
    args.time_steps = 1000
    model = Unet3(args)
    x = torch.randn(1, 3, 128, 128)
    t = torch.randint(0, 1000, (1,))
    y = model(x, t)
    loss = F.mse_loss(y,x)
    loss.backward()
    print(loss)
    print(y.shape)
