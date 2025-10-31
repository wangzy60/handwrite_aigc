import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import logging


def is_all_zeros(tensor, tol=1e-6):
    """检查张量是否全零（考虑浮点误差）"""
    return torch.all(torch.abs(tensor) < tol).item()

def init_ddpm_params(model):
    if isinstance(model, (nn.Conv2d, nn.Linear)):
        if is_all_zeros(model.weight):
            logging.info(f"跳过全零初始化层: {model}")
        else:
            nn.init.kaiming_normal_(model.weight, mode='fan_in', nonlinearity="relu")
        if model.bias is not None:
            nn.init.zeros_(model.bias)
    elif isinstance(model, nn.Embedding):
        nn.init.kaiming_normal_(model.weight, mode='fan_in', nonlinearity="relu")
    elif isinstance(model, nn.LayerNorm):
        nn.init.ones_(model.weight)
        nn.init.zeros_(model.bias)


def init_vae_params(model):
    if isinstance(model, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(model.weight, mode='fan_in', nonlinearity="relu")
        if model.bias is not None:
            nn.init.zeros_(model.bias)
    elif isinstance(model, nn.Embedding):
        nn.init.kaiming_normal_(model.weight, mode='fan_in', nonlinearity="relu")
    elif isinstance(model, nn.LayerNorm):
        nn.init.ones_(model.weight)
        nn.init.zeros_(model.bias)


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


class Time_Encode(nn.Module):
    def __init__(self, embedding_dims, h, out_dims, activate_func_name):
        #参数量较多，可能设计过度，在这个小模型里可能不能充分的利用到所有参数
        super().__init__()
        self.time_embedding = nn.Sequential(nn.Linear(embedding_dims, int(h**2)),
                                            get_activate_func(activate_func_name))
        self.conv = nn.Conv2d(1, out_dims, kernel_size=1, stride=1)
        self.h = int(h)
            
    def forward(self, t):
        x = self.time_embedding(t)
        x = x.view(-1, 1, self.h, self.h)
        x = self.conv(x)
        return x



class Encode_Block(nn.Module):
    def __init__(self, in_channel, out_channel, image_size, activate_func_name, kernel_size=3, stride=1, padding=1):
        #encoder部分是否要进行残差连接？
        super().__init__()
        self.conv = nn.Sequential(nn.LayerNorm([in_channel, int(image_size), int(image_size)]),
                                  nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),
                                  get_activate_func(activate_func_name),
                                  nn.Conv2d(out_channel, out_channel, kernel_size, stride, padding),
                                  get_activate_func(activate_func_name))
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x, t):
        # x(batch, channel, height, width)
        # t(batch, channel, height, width)
        x = x+t
        x_skip = self.conv(x)
        x = self.downsample(x_skip)
        return x_skip, x

class Decode_Block(nn.Module):
    def __init__(self, in_channel, out_channel, skip_image_size, activate_func_name, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding))
        self.conv = nn.Sequential(nn.LayerNorm([in_channel, int(skip_image_size), int(skip_image_size)]),
                                  nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),
                                  get_activate_func(activate_func_name),
                                  nn.Conv2d(out_channel, out_channel, kernel_size, stride, padding),
                                  get_activate_func(activate_func_name))
    
    def forward(self, x, x_skip, t):
        # x(batch, channel, height, width)
        # x_skip(batch, channel, height, width)
        # t(batch, channel, height, width)
        x = self.upsample(x)
        x = torch.concat([x, x_skip], dim=1)
        x = x + t
        return self.conv(x)

class Unet1(nn.Module):
    def __init__(self, args):
        super().__init__()
        h = args.input_image_size
        c_in = args.input_image_dims
        c_out = args.output_image_dims
        time_dims = args.time_embedding_dims
        T = args.time_steps
        activate_func_name = args.train_activate_func

        self.time_embedding = nn.Embedding(T, time_dims)
        self.time_encoder_1 = Time_Encode(time_dims, h, c_in, activate_func_name)
        self.image_encoder_1 = Encode_Block(c_in, 64, h, activate_func_name)
        self.time_encoder_2 = Time_Encode(time_dims, h/2, 64, activate_func_name)
        self.image_encoder_2 = Encode_Block(64, 128, h/2, activate_func_name)
        self.time_encoder_3 = Time_Encode(time_dims, h/4, 128, activate_func_name)
        self.image_encoder_3 = Encode_Block(128, 256, h/4, activate_func_name)
        self.time_encoder_4 = Time_Encode(time_dims, h/8, 256, activate_func_name)
        self.image_encoder_4 = Encode_Block(256, 512, h/8, activate_func_name)
        self.time_bottle_neck = Time_Encode(time_dims, h/16, 512, activate_func_name)
        self.image_boottle_neck = nn.Sequential(nn.LayerNorm([512, int(h/16), int(h/16)]),
                                                nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
                                                get_activate_func(activate_func_name),
                                                nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
                                                get_activate_func(activate_func_name))
        self.time_decoder_4 = Time_Encode(time_dims, h/8, 1024, activate_func_name)
        self.image_decoder_4 = Decode_Block(1024, 512, h/8, activate_func_name)
        self.time_decoder_3 = Time_Encode(time_dims, h/4, 512, activate_func_name)
        self.image_decoder_3 = Decode_Block(512, 256, h/4, activate_func_name)
        self.time_decoder_2 = Time_Encode(time_dims, h/2, 256, activate_func_name)
        self.image_decoder_2 = Decode_Block(256, 128, h/2, activate_func_name)
        self.time_decoder_1 = Time_Encode(time_dims, h, 128, activate_func_name)
        self.image_decoder_1 = Decode_Block(128, 64, h, activate_func_name)
        self.output = nn.Sequential(nn.LayerNorm([64, int(h), int(h)]),
                                    nn.Conv2d(64, c_out, kernel_size=1, stride=1))
        
        #初始化所有网络参数
        self.apply(init_ddpm_params)

    def forward(self, x, t):
        # x(batch, channel, height, width)
        # t(batch, time_index)
        assert x.shape[0] == t.shape[0], "Batch size of input and target must be same"
        t = self.time_embedding(t-1)
        e_t1 = self.time_encoder_1(t)
        e_t2 = self.time_encoder_2(t)
        e_t3 = self.time_encoder_3(t)
        e_t4 = self.time_encoder_4(t)
        t_b = self.time_bottle_neck(t)
        d_t4 = self.time_decoder_4(t)
        d_t3 = self.time_decoder_3(t)
        d_t2 = self.time_decoder_2(t)
        d_t1 = self.time_decoder_1(t)
        
        x_skip_1, x = self.image_encoder_1(x, e_t1)
        x_skip_2, x = self.image_encoder_2(x, e_t2)
        x_skip_3, x = self.image_encoder_3(x, e_t3)
        x_skip_4, x = self.image_encoder_4(x, e_t4)

        x = x + t_b
        x = self.image_boottle_neck(x)

        x = self.image_decoder_4(x, x_skip_4, d_t4)
        x = self.image_decoder_3(x, x_skip_3, d_t3)
        x = self.image_decoder_2(x, x_skip_2, d_t2)
        x = self.image_decoder_1(x, x_skip_1, d_t1)
        return self.output(x)
    
class conv_block(nn.Module):
    def __init__(self, input_dim, out_dim, image_size, activate_func_name):
        super().__init__()
        if input_dim == 3:
            self.conv = nn.Sequential(nn.Conv2d(input_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                    get_activate_func(activate_func_name))
        else:
            self.conv = nn.Sequential(nn.GroupNorm(num_groups=32, num_channels=input_dim),
                                    nn.Conv2d(input_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                    get_activate_func(activate_func_name))
    
    def forward(self, x):
        return self.conv(x)


class conv_first_block(nn.Module):
    def __init__(self, input_dim, out_dim, image_size, activate_func_name):
        super().__init__()
        if input_dim >= 32:
            self.conv = nn.Sequential(nn.GroupNorm(num_groups=32, num_channels=input_dim),
                                      nn.Conv2d(input_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                      get_activate_func(activate_func_name))
        else:
            self.conv = nn.Sequential(nn.Conv2d(input_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                      get_activate_func(activate_func_name))            
    
    def forward(self, x):
        return self.conv(x)


class conv_block_acti_func_forward(nn.Module):
    def __init__(self, input_dim, out_dim, image_size, activate_func_name):
        super().__init__()
        if input_dim < 32:
            self.conv = nn.Sequential(nn.Conv2d(input_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                      get_activate_func(activate_func_name))
        else:
            self.conv = nn.Sequential(nn.GroupNorm(num_groups=32, num_channels=input_dim),
                                      get_activate_func(activate_func_name),
                                      nn.Conv2d(input_dim, out_dim, kernel_size=3, stride=1, padding=1))
    
    def forward(self, x):
        return self.conv(x)


class downsampling(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        return self.downsample(x)
    
class maxpool_downsampling(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool_downsample = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        return self.maxpool_downsample(x)

class conv_downsampling(nn.Module):
    def __init__(self, input_channel, activate_func_name):
        super().__init__()
        self.activate_func_name = activate_func_name
        self.conv = nn.Sequential(nn.GroupNorm(num_groups=32, num_channels=input_channel),
                                  nn.Conv2d(input_channel, input_channel, kernel_size=3, stride=2, padding=1),
                                  get_activate_func(activate_func_name))

    def forward(self, x):
        return self.conv(x)
    
class vae_conv_downsampling(nn.Module):
    def __init__(self, input_channel, activate_func_name):
        super().__init__()
        self.activate_func_name = activate_func_name
        self.conv = nn.Sequential(nn.GroupNorm(num_groups=32, num_channels=input_channel),
                                  get_activate_func(activate_func_name),
                                  nn.Conv2d(input_channel, input_channel, kernel_size=3, stride=2, padding=1))

    def forward(self, x):
        return self.conv(x)


class upsampling(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
        self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                      nn.Conv2d(input_channel, input_channel, kernel_size=3, stride=1, padding=1))
    
    def forward(self, x):
        return self.upsample(x)
    

class bilinear_upsampling_block(nn.Module):
    def __init__(self, input_channel, activate_func_name):
        super().__init__()
        self.activate_func_name = activate_func_name
        self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                      nn.GroupNorm(num_groups=32, num_channels=input_channel),
                                      nn.Conv2d(input_channel, input_channel, kernel_size=3, stride=1, padding=1),
                                      get_activate_func(activate_func_name))
    
    def forward(self, x):
        return self.upsample(x)
    

class vae_bilinear_upsampling_block(nn.Module):
    def __init__(self, input_channel, activate_func_name):
        super().__init__()
        self.activate_func_name = activate_func_name
        self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                      nn.GroupNorm(num_groups=32, num_channels=input_channel),
                                      get_activate_func(activate_func_name),
                                      nn.Conv2d(input_channel, input_channel, kernel_size=3, stride=1, padding=1))
    
    def forward(self, x):
        return self.upsample(x)

    
class residual_block(nn.Module):
    def __init__(self, input_dim, out_dim, image_size, activate_func_name):
        super().__init__()
        self.activate_func_name = activate_func_name
        self.conv = nn.Sequential(nn.GroupNorm(num_groups=32, num_channels=input_dim),
                                  nn.Conv2d(input_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                  get_activate_func(activate_func_name),
                                  nn.GroupNorm(num_groups=32, num_channels=out_dim),
                                  nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                  get_activate_func(activate_func_name))
        self.resdital = nn.Identity() if input_dim == out_dim else nn.Conv2d(input_dim, out_dim, kernel_size=1, stride=1)

    def forward(self, x):
        return self.conv(x) + self.resdital(x)


def set_module_zero(module):
    for param in module.parameters():
        param.detach().zero_()
    return module

class residual_block_with_time_embedding(nn.Module):
    def __init__(self, input_dim, out_dim, time_embedding_dim, activate_func_name):
        super().__init__()
        self.activate_func_name = activate_func_name
        self.conv1 = nn.Sequential(nn.GroupNorm(num_groups=32, num_channels=input_dim),
                                   get_activate_func(activate_func_name),
                                   nn.Conv2d(input_dim, out_dim, kernel_size=3, stride=1, padding=1))
        self.time = nn.Sequential(nn.Linear(time_embedding_dim, out_dim), 
                                  get_activate_func(activate_func_name))
        self.conv2 = nn.Sequential(nn.GroupNorm(num_groups=32, num_channels=out_dim),
                                   get_activate_func(activate_func_name),
                                   set_module_zero(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)))
        self.resdital = nn.Identity() if input_dim == out_dim \
                                      else nn.Conv2d(input_dim, out_dim, kernel_size=1, stride=1)

    def forward(self, x, time_embedding):
        input_x = x
        x = self.conv1(x)
        time = self.time(time_embedding).squeeze()
        while len(time.shape) < len(x.shape):
            time = time[..., None]
        x = x + time
        x = self.conv2(x)
        return self.resdital(input_x) + x


class residual_block_acti_func_forward(nn.Module):
    def __init__(self, input_dim, out_dim, image_size, activate_func_name):
        super().__init__()
        self.activate_func_name = activate_func_name
        self.conv = nn.Sequential(nn.GroupNorm(num_groups=32, num_channels=input_dim),
                                  get_activate_func(activate_func_name),
                                  nn.Conv2d(input_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                  nn.GroupNorm(num_groups=32, num_channels=out_dim),
                                  get_activate_func(activate_func_name),
                                  nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1))
        self.resdital = nn.Identity() if input_dim == out_dim else nn.Conv2d(input_dim, out_dim, kernel_size=1, stride=1)

    def forward(self, x):
        return self.conv(x) + self.resdital(x)


class residual_block_with_cross_attation(nn.Module):
    def __init__(self, input_dim, out_dim, text_embedding_dim, image_size, activate_func_name):
        super().__init__()
        self.activate_func_name = activate_func_name
        self.conv1 = nn.Sequential(nn.GroupNorm(num_groups=32, num_channels=input_dim),
                                  nn.Conv2d(input_dim, out_dim, kernel_size=3, stride=1, padding=1))
        self.conv2 = nn.Sequential(nn.GroupNorm(num_groups=32, num_channels=out_dim),
                                  nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1))
        self.attation = image_text_multihead_cross_attation_block(out_dim, text_embedding_dim, text_embedding_dim, out_dim, out_dim)
        self.activation =  get_activate_func(activate_func_name)
        self.resdital = nn.Identity() if input_dim == out_dim else nn.Conv2d(input_dim, out_dim, kernel_size=1, stride=1)

    def forward(self, input_img, txt_embedding):
        img = self.conv1(input_img)
        img = self.conv2(img)
        img = self.attation(img, txt_embedding, txt_embedding)
        img = self.activation(img)
        return self.resdital(input_img) + img

    
class time_embedding_block(nn.Module):
    def __init__(self, input_dim, out_dim, image_size, activate_func_name):
        super().__init__()
        self.image_size = image_size
        self.activate_func_name = activate_func_name
        self.time = nn.Sequential(nn.Linear(input_dim, out_dim),
                                  get_activate_func(activate_func_name))
    
    def forward(self, x):
        # x(batch, input_dim) -> (batch, image_size, image_size, out_dim)
        x = self.time(x).squeeze().unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.image_size, self.image_size)
        return x


class multihead_attation_block(nn.Module):
    def __init__(self,q_in_dim, k_in_dim, v_in_dim, latent_dim, out_dim, heads = 8):
        #这个版本实现有问题，将图像的长宽当做了注意力的维度，可能会导致问题。
        #更新后的实现见image_multihead_self_attation_block类
        super().__init__()
        self.heads = heads
        self.latent_dim = latent_dim
        assert latent_dim % heads == 0, f"multi head latent dim must be divisible by heads number, but got latent dim: {latent_dim} and heads: {heads}"
        self.q_to_latent = nn.Linear(q_in_dim, latent_dim)
        self.k_to_latent = nn.Linear(k_in_dim, latent_dim)
        self.v_to_latent = nn.Linear(v_in_dim, latent_dim)
        self.latent_to_out = nn.Linear(latent_dim, out_dim)

    def forward(self, q,k=None,v=None):
        # q/k/v: (batch, channel, height*width) or (batch, channel, height, width)
        heads = self.heads
        latent_dim = self.latent_dim
        if k is None:
            k = q
        if v is None:
            v = k
        q_shape = q.shape
        q = q.flatten(start_dim=2)
        k = k.flatten(start_dim=2)
        v = v.flatten(start_dim=2)
        q = self.q_to_latent(q)
        k = self.k_to_latent(k)
        v = self.v_to_latent(v)
        batch_size, channel, _ = q.shape
        q = q.reshape(batch_size, channel, heads, latent_dim//heads).permute(0, 2, 1, 3)
        k = k.reshape(batch_size, channel, heads, latent_dim//heads).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, channel, heads, latent_dim//heads).permute(0, 2, 1, 3)
        attation_score = torch.matmul(q, k.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(latent_dim//heads))
        attation_score = F.softmax(attation_score, dim = -1)
        x = torch.matmul(attation_score, v)
        x = x.permute(0, 2, 1, 3).reshape(batch_size, channel, -1)
        x = self.latent_to_out(x).reshape(*q_shape)
        return x


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
        # attation_score = torch.matmul(q, k.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(latent_dim//heads)) # b, head, h*w, h*w
        # attation_score = F.softmax(attation_score, dim = -1) # b, head, h*w, h*w
        # x = torch.matmul(attation_score, v) # b, head, h*w, h*w @ b, head, h*w, c//head -> b, head, h*w, c//head
        
        x = x.permute(0, 2, 1, 3).contiguous().reshape(batch_size, seq, -1) # b, head, h*w, c//head -> b, h*w, head, c//head -> b, h*w, c
        x = self.latent_to_out(x).permute(0,2,1).contiguous().reshape(*image_shape) #b, h*w, c -> b, c, h*w -> b, c, h, w
        return input_x + x  
    
    def forward(self, x):
        return checkpoint(self._forward, x, use_reentrant=False)


class image_multihead_attation_block(nn.Module):
    def __init__(self,q_in_dim, k_in_dim, v_in_dim, latent_dim, out_dim, heads = 8):
        #仅支持使用图像作为注意力的输入
        super().__init__()
        self.heads = heads
        self.latent_dim = latent_dim
        assert latent_dim % heads == 0, f"multi head latent dim must be divisible by heads number, but got latent dim: {latent_dim} and heads: {heads}"
        self.q_norm = nn.GroupNorm(num_groups=32, num_channels=q_in_dim)
        self.k_norm = nn.GroupNorm(num_groups=32, num_channels=k_in_dim)
        self.v_norm = nn.GroupNorm(num_groups=32, num_channels=v_in_dim)
        self.q_to_latent = nn.Linear(q_in_dim, latent_dim)
        self.k_to_latent = nn.Linear(k_in_dim, latent_dim)
        self.v_to_latent = nn.Linear(v_in_dim, latent_dim)
        self.latent_to_out = nn.Linear(latent_dim, out_dim)

    def forward(self, q,k=None,v=None):
        # q/k/v: (batch, channel, height*width) or (batch, channel, height, width)
        heads = self.heads
        latent_dim = self.latent_dim
        if k is None:
            k = q
        if v is None:
            v = k
            
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        
        v_shape = v.shape
        q = q.flatten(start_dim=2).permute(0, 2, 1) # b, c, h, w -> b, h*w, c
        k = k.flatten(start_dim=2).permute(0, 2, 1)
        v = v.flatten(start_dim=2).permute(0, 2, 1)
        q = self.q_to_latent(q)
        k = self.k_to_latent(k)
        v = self.v_to_latent(v)
        batch_size, seq, _ = q.shape
        q = q.reshape(batch_size, seq, heads, latent_dim//heads).permute(0, 2, 1, 3) # b, h*w, c -> b, h*w, head, c//head -> b, head, h*w, c//head
        k = k.reshape(batch_size, seq, heads, latent_dim//heads).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, seq, heads, latent_dim//heads).permute(0, 2, 1, 3)
        #需要注意，torch.matmul在进行大于等于3维的矩阵乘法时，只有最后两个维度参与矩阵乘法，其他维度不参与矩阵乘法
        #具体来说，对于A（2,3,4）和B（2,4,3）两个矩阵，使用torch.matmul进行计算，得到的结果的形状为C（2,3,3）
        #这个计算是分两步得到的，首先算C[0] = A[0]@B[0]，然后计算C[1] = A[1]@B[1]，然后得到C
        #如果是4维矩阵，例如（2,3,4,5），则2和3所在的维度都不参与矩阵乘法，依此类推
        attation_score = torch.matmul(q, k.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(latent_dim//heads)) # b, head, h*w, h*w
        attation_score = F.softmax(attation_score, dim = -1) # b, head, h*w, h*w
        x = torch.matmul(attation_score, v) # b, head, h*w, h*w @ b, head, h*w, c//head -> b, head, h*w, c//head
        x = x.permute(0, 2, 1, 3).reshape(batch_size, seq, -1) # b, head, h*w, c//head -> b, h*w, head, c//head -> b, h*w, c
        x = self.latent_to_out(x).permute(0,2,1).reshape(*v_shape) #b, h*w, c -> b, c, h*w -> b, c, h, w
        return x  #这里忘记与输入进行残差连接了，加上残差连接效果可能更好


class image_text_multihead_cross_attation_block(nn.Module):
    def __init__(self,q_in_dim, k_in_dim, v_in_dim, latent_dim, out_dim, heads = 8):
        #支持使用图像作为注意力的query，文本作为注意力的key和value
        super().__init__()
        self.heads = heads
        self.latent_dim = latent_dim
        assert latent_dim % heads == 0, f"multi head latent dim must be divisible by heads number, but got latent dim: {latent_dim} and heads: {heads}"
        self.q_norm = nn.GroupNorm(num_groups=32, num_channels=q_in_dim)
        self.k_norm = nn.LayerNorm(k_in_dim)
        self.v_norm = nn.LayerNorm(v_in_dim)
        self.q_to_latent = nn.Linear(q_in_dim, latent_dim)
        self.k_to_latent = nn.Linear(k_in_dim, latent_dim)
        self.v_to_latent = nn.Linear(v_in_dim, latent_dim)
        self.latent_to_out = nn.Linear(latent_dim, out_dim)

    def forward(self, q,k,v):
        # q: img (batch, channel, height, width)
        # kv:text(batch, text_seq, text_emb)
        heads = self.heads
        latent_dim = self.latent_dim
        input_img = q
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        q_shape = q.shape
        q = q.flatten(start_dim=2).permute(0, 2, 1) # b, c, h, w -> b, h*w, c
        q = self.q_to_latent(q)
        k = self.k_to_latent(k)
        v = self.v_to_latent(v)
        batch_size, img_seq, _ = q.shape
        _, text_seq, _ = k.shape
        q = q.reshape(batch_size, img_seq, heads, latent_dim//heads).permute(0, 2, 1, 3) # b, img_seq, latent_dim -> b, img_seq, head, latent_dim//head -> b, head, img_seq, latent_dim//head
        k = k.reshape(batch_size, text_seq, heads, latent_dim//heads).permute(0, 2, 1, 3) # b, text_seq, latent_dim -> b, text_seq, head, latent_dim//head -> b, head, text_seq, latent_dim//head
        v = v.reshape(batch_size, text_seq, heads, latent_dim//heads).permute(0, 2, 1, 3)
        attation_score = torch.matmul(q, k.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(latent_dim//heads)) # b, head, img_seq, latent_dim//head @ b, head, latent_dim//head, text_seq -> b, head, img_seq, text_seq
        attation_score = F.softmax(attation_score, dim = -1) # b, head, img_seq, text_seq
        x = torch.matmul(attation_score, v) # b, head, img_seq, text_seq @ b, head, text_seq, latent_dim//head -> b, head, img_seq, latent_dim//head
        x = x.permute(0, 2, 1, 3).reshape(batch_size, img_seq, -1) # b, head, img_seq, latent_dim//head -> b, img_seq, head, latent_dim//head -> b, img_seq, latent_dim
        x = self.latent_to_out(x).permute(0,2,1).reshape(*q_shape) #b, img_seq, latent_dim -> b, img_seq, out_dim -> b, out_dim, img_seq -> b, c, h, w
        return input_img + x




class Unet3_res_blocks(nn.Module):
    def __init__(self, c_in, c_out, time_dims, activate_func_name, input_img_h, downsample = False, upsample = False, use_attation = False):
        super().__init__()
        self.downsample = downsample
        self.upsample = upsample
        self.use_attation = use_attation
        assert not (self.downsample and self.upsample), "不能同时进行下采样和上采样"
        self.res1 = residual_block_with_time_embedding(c_in, c_out, time_dims, activate_func_name)
        self.res2 = residual_block_with_time_embedding(c_out, c_out, time_dims, activate_func_name)
        if use_attation:
            self.self_attation = image_slef_attation_block(c_out, activate_func_name)
        self.res3 = residual_block_with_time_embedding(c_out, c_out, time_dims, activate_func_name)
        if self.downsample:
            self.updown = conv_downsampling(c_out, activate_func_name)  #如果是下采样，下采样放在模块的最后，所以channel是c_out
        elif self.upsample:
            self.updown = bilinear_upsampling_block(c_in//2, activate_func_name)  #如果是上采样，上采样放在模块的最前，所以channel是c_in
        else:
            self.updown = None

    def forward(self, x, time_embedding, res=None):
        if self.upsample:
            assert res is not None, "Unet上采样过程中，未与下采样的输出进行连接，请检查"
            x = self.updown(x)
            x = torch.concat((x, res), dim=1)
        x = self.res1(x, time_embedding)
        x = self.res2(x, time_embedding)
        if self.use_attation:
            x = self.self_attation(x)
        res = self.res3(x, time_embedding)
        if self.downsample:
            x = self.updown(res)
        else:
            x = res
        return x, res