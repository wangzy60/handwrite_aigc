import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import numpy as np

def get_activate_func(activate_func_name):
    if activate_func_name == 'relu':
        return nn.ReLU()
    elif activate_func_name == 'gelu':
        return nn.GELU()
    elif activate_func_name == 'silu':
        return nn.SiLU()
    elif activate_func_name == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError(f"Not support this activation function {activate_func_name}")    

class image_slef_attation_block(nn.Module):
    def __init__(self,in_dim, activate_func_name, heads = 8):
        #仅支持使用图像作为注意力的输入
        super().__init__()
        self.heads = heads
        self.in_dim = in_dim
        assert in_dim % heads == 0, f"multi head latent dim must be divisible by heads number, but got latent dim: {in_dim} and heads: {heads}"
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_dim)
        self.qkv_proj = nn.Linear(in_dim, 3*in_dim)
        self.latent_to_out = nn.Sequential(nn.Linear(in_dim, in_dim),
                                           get_activate_func(activate_func_name))
                                           
    def _forward(self, x):
        # q/k/v: (batch, channel, height*width) or (batch, channel, height, width)
        # permute后显示使用contiguous避免内存不连续问题
        input_x = x
        heads = self.heads
        latent_dim = self.in_dim
        x = self.norm(x)
        image_shape = x.shape
        x = x.flatten(start_dim=2).permute(0, 2, 1).contiguous() # b, c, h, w -> b, h*w, c
        x = self.qkv_proj(x)
        q,k,v = x.chunk(3, dim=-1)
        batch_size, seq, _ = q.shape
        q = q.reshape(batch_size, seq, heads, latent_dim//heads).permute(0, 2, 1, 3).contiguous() # b, h*w, c -> b, h*w, head, c//head -> b, head, h*w, c//head
        k = k.reshape(batch_size, seq, heads, latent_dim//heads).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(batch_size, seq, heads, latent_dim//heads).permute(0, 2, 1, 3).contiguous()
        
        #使用torch的注意力计算减少显存占用，默认会除以根号d，不用手动设置
        x = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask = None,
            dropout_p = 0.0,
            is_causal = False)
        
        #自己实现注意力计算
        attation_score = torch.matmul(q, k.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(latent_dim//heads)) # b, head, h*w, h*w
        attation_score = F.softmax(attation_score, dim = -1) # b, head, h*w, h*w
        x1 = torch.matmul(attation_score, v) # b, head, h*w, h*w @ b, head, h*w, c//head -> b, head, h*w, c//head
        
        print(f"offical: {x}")
        print(f"selmade: {x1}")
        print(f"gap: {torch.sum(torch.abs(x1 - x) > 1e-6)}")
        print(f"gap: {torch.sum(torch.abs(x1 - x))}")
        
        x = x.permute(0, 2, 1, 3).contiguous().reshape(batch_size, seq, -1) # b, head, h*w, c//head -> b, h*w, head, c//head -> b, h*w, c
        x = self.latent_to_out(x).permute(0,2,1).contiguous().reshape(*image_shape) #b, h*w, c -> b, c, h*w -> b, c, h, w
        return input_x + x  
    
    def forward(self, x):
        return checkpoint(self._forward, x, use_reentrant=False)


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    act_name = "silu"
    in_dim = 1024
    img_size = 64
    attation_block = image_slef_attation_block(in_dim, act_name)
    x = torch.randn(1,in_dim, img_size, img_size)
    attation_block(x)
