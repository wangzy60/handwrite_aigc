import torch
import torch.nn as nn
import torch.nn.functional as F



def init_ddpm_params(model):
    if isinstance(model, (nn.Conv2d, nn.Linear)):
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

class Oriented_Conv(nn.Module):
    def __init__(self, input_dim, out_dim, stride = 1):
        super().__init__()
        self.hidden_dim = max(out_dim//4, 1)
        self.input_dim = input_dim
        if self.input_dim % 32 == 0:
            self.gn = nn.GroupNorm(num_groups=32, num_channels=input_dim)
        self.conv_0 = nn.Sequential(nn.SiLU(),
                                    nn.Conv2d(input_dim, self.hidden_dim, kernel_size=3, stride = stride, padding = (1, 4), dilation=(1,4)))
        self.conv_45 = nn.Sequential(nn.SiLU(),
                                    nn.Conv2d(input_dim, self.hidden_dim, kernel_size=3, stride = stride, padding = (2, 2), dilation=(2,2)))
        self.conv_90 = nn.Sequential(nn.SiLU(),
                                    nn.Conv2d(input_dim, self.hidden_dim, kernel_size=3, stride = stride, padding = (4, 1), dilation=(4,1)))
        self.conv_135 = nn.Sequential(nn.SiLU(),
                                    nn.Conv2d(input_dim, self.hidden_dim, kernel_size=3, stride = stride, padding = (4, 4), dilation=(4,4)))
        self.out = nn.Conv2d(self.hidden_dim*4, out_dim, kernel_size=1)
    
    def forward(self, x):
        if self.input_dim % 32 == 0:
            x = self.gn(x)
        return self.out(torch.cat([self.conv_0(x),
                                   self.conv_45(x),
                                   self.conv_90(x),
                                   self.conv_135(x)], dim = 1))

class oriented_conv_block(nn.Module):
    def __init__(self, input_dim, out_dim, image_size, activate_func_name):
        super().__init__()
        if input_dim < 32:
            self.conv = nn.Sequential(Oriented_Conv(input_dim, out_dim),
                                      get_activate_func(activate_func_name))
        else:
            self.conv = nn.Sequential(nn.GroupNorm(num_groups=32, num_channels=input_dim),
                                      get_activate_func(activate_func_name),
                                      Oriented_Conv(input_dim, out_dim))
    
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
                                  get_activate_func(activate_func_name),
                                  nn.Conv2d(input_channel, input_channel, kernel_size=3, stride=2, padding=1))

    def forward(self, x):
        return self.conv(x)


class oriented_conv_downsampling(nn.Module):
    def __init__(self, input_channel, activate_func_name):
        super().__init__()
        self.activate_func_name = activate_func_name
        self.conv = nn.Sequential(nn.GroupNorm(num_groups=32, num_channels=input_channel),
                                  get_activate_func(activate_func_name),
                                  Oriented_Conv(input_channel, input_channel, stride=2))

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
                                      get_activate_func(activate_func_name),
                                      nn.Conv2d(input_channel, input_channel, kernel_size=3, stride=1, padding=1))
    
    def forward(self, x):
        return self.upsample(x)

class oriented_bilinear_upsampling_block(nn.Module):
    def __init__(self, input_channel, activate_func_name):
        super().__init__()
        self.activate_func_name = activate_func_name
        self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                      nn.GroupNorm(num_groups=32, num_channels=input_channel),
                                      get_activate_func(activate_func_name),
                                      Oriented_Conv(input_channel, input_channel))
    
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


class oriented_residual_block(nn.Module):
    def __init__(self, input_dim, out_dim, image_size, activate_func_name):
        super().__init__()
        self.activate_func_name = activate_func_name
        self.conv = nn.Sequential(nn.GroupNorm(num_groups=32, num_channels=input_dim),
                                  get_activate_func(activate_func_name),
                                  Oriented_Conv(input_dim, out_dim),
                                  nn.GroupNorm(num_groups=32, num_channels=out_dim),
                                  get_activate_func(activate_func_name),
                                  Oriented_Conv(out_dim, out_dim))
        self.resdital = nn.Identity() if input_dim == out_dim else nn.Conv2d(input_dim, out_dim, kernel_size=1, stride=1)

    def forward(self, x):
        return self.conv(x) + self.resdital(x)
    

class residual_block_with_cross_attation(nn.Module):
    def __init__(self, input_dim, out_dim, text_embedding_dim, image_size, activate_func_name):
        super().__init__()
        self.activate_func_name = activate_func_name
        self.conv1 = nn.Sequential(nn.GroupNorm(num_groups=32, num_channels=input_dim),
                                  get_activate_func(activate_func_name),
                                  nn.Conv2d(input_dim, out_dim, kernel_size=3, stride=1, padding=1))
        self.attation = image_text_multihead_cross_attation_block(out_dim, text_embedding_dim, text_embedding_dim, out_dim, out_dim)
        self.con2 = nn.Sequential(nn.GroupNorm(num_groups=32, num_channels=out_dim),
                                  get_activate_func(activate_func_name),
                                  nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1))
        self.resdital = nn.Identity() if input_dim == out_dim else nn.Conv2d(input_dim, out_dim, kernel_size=1, stride=1)

    def forward(self, input_img, txt_embedding):
        img = self.conv1(input_img)
        img = self.attation(img, txt_embedding, txt_embedding)
        img = self.conv2(img)
        return input_img + self.resdital(img)

    
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
    
class Unet2(nn.Module):
    def __init__(self, args):
        super().__init__()
        h = args.input_image_size
        c_in = args.input_image_dims
        c_out = args.output_image_dims
        time_dims = args.time_embedding_dims
        activate_func_name = args.train_activate_func
        T = args.time_steps
        
        self.time_embedding = nn.Embedding(T, time_dims)
        
        self.conv1 = conv_block(c_in, 64, int(h), activate_func_name)
        self.time1 = time_embedding_block(time_dims, 64, int(h), activate_func_name)
        self.res1 = residual_block(64, 64, int(h), activate_func_name)
        
        self.downsample1 = downsampling()
        
        self.conv2 = conv_block(64, 128, int(h/2), activate_func_name)
        self.time2 = time_embedding_block(time_dims, 128, int(h/2), activate_func_name)
        self.res2 = residual_block(128, 128, int(h/2), activate_func_name)
        
        self.downsample2 = downsampling()
        
        self.conv3 = conv_block(128, 256, int(h/4), activate_func_name)
        self.time3 = time_embedding_block(time_dims, 256, int(h/4), activate_func_name)
        self.res3 = residual_block(256, 256, int(h/4), activate_func_name)
        
        self.downsample3 = downsampling()
        
        self.conv4 = conv_block(256, 512, int(h/8), activate_func_name)
        self.time4 = time_embedding_block(time_dims, 512, int(h/8), activate_func_name)
        self.res4 = residual_block(512, 512, int(h/8), activate_func_name)
        
        self.downsample4 = downsampling()
        
        self.bottleneckle_conv = conv_block(512, 1024, int(h/16), activate_func_name)
        self.bottleneckle_time = time_embedding_block(time_dims, 1024, int(h/16), activate_func_name)
        self.bottleneckle_res = residual_block(1024, 1024, int(h/16), activate_func_name)
        
        self.upsample5 = upsampling(1024)
        
        self.conv5 = conv_block(1024 + 512, 768, int(h/8), activate_func_name)
        self.time5 = time_embedding_block(time_dims, 768, int(h/8), activate_func_name)
        self.res5 = residual_block(768, 768, int(h/8), activate_func_name)
        
        self.upsample6 = upsampling(768)
        
        self.conv6 = conv_block(768 + 256, 512, int(h/4), activate_func_name)
        self.time6 = time_embedding_block(time_dims, 512, int(h/4), activate_func_name)
        self.res6 = residual_block(512, 512, int(h/4), activate_func_name)
        
        self.upsample7 = upsampling(512)
        
        self.conv7 = conv_block(512 + 128, 320, int(h/2), activate_func_name)
        self.time7 = time_embedding_block(time_dims, 320, int(h/2), activate_func_name)
        self.res7 = residual_block(320, 320, int(h/2), activate_func_name)
        
        self.upsample8 = upsampling(320)
        
        self.conv8 = conv_block(320 + 64, 192, int(h), activate_func_name)
        self.time8 = time_embedding_block(time_dims, 192, int(h), activate_func_name)
        self.res8 = residual_block(192, 192, int(h), activate_func_name)
        
        self.output = nn.Sequential(nn.GroupNorm(num_groups=32, num_channels=192),
                                    nn.Conv2d(192, 96, kernel_size=1, stride=1),
                                    get_activate_func(activate_func_name),
                                    nn.GroupNorm(num_groups=32, num_channels=96),
                                    nn.Conv2d(96, 32, kernel_size=1, stride=1),
                                    get_activate_func(activate_func_name),
                                    nn.Conv2d(32, c_out, kernel_size=1, stride=1))
        
        self.apply(init_ddpm_params)
        
    def forward(self, x, t):
        t = self.time_embedding(t-1)
        t_1 = self.time1(t)
        t_2 = self.time2(t)
        t_3 = self.time3(t)
        t_4 = self.time4(t)
        t_bn = self.bottleneckle_time(t)
        t_5 = self.time5(t)
        t_6 = self.time6(t)
        t_7 = self.time7(t)
        t_8 = self.time8(t)
        
        x = self.conv1(x)
        x = x + t_1
        x_skp1 = self.res1(x)
        x = self.downsample1(x_skp1)
        
        x = self.conv2(x)
        x = x + t_2
        x_skp2 = self.res2(x)
        x = self.downsample2(x_skp2)
        
        x = self.conv3(x)
        x = x + t_3
        x_skp3 = self.res3(x)
        x = self.downsample3(x_skp3)
        
        x = self.conv4(x)
        x = x + t_4
        x_skp4 = self.res4(x)
        x = self.downsample3(x_skp4)
        
        x = self.bottleneckle_conv(x)
        x = x + t_bn
        x = self.bottleneckle_res(x)
        
        x = self.upsample5(x)
        x = torch.cat((x, x_skp4), dim=1)
        x = self.conv5(x)
        x = x + t_5
        x = self.res5(x)
        
        x = self.upsample6(x)
        x = torch.cat((x, x_skp3), dim=1)
        x = self.conv6(x)
        x = x + t_6
        x = self.res6(x)
        
        x = self.upsample7(x)
        x = torch.cat((x, x_skp2), dim=1)
        x = self.conv7(x)
        x = x + t_7
        x = self.res7(x)
        
        x = self.upsample8(x)
        x = torch.cat((x, x_skp1), dim=1)
        x = self.conv8(x)
        x = x + t_8
        x = self.res8(x)
        
        x = self.output(x)
        
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
        attation_score = torch.matmul(q, k.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(latent_dim//heads)) # b, head, h*w, h*w
        attation_score = F.softmax(attation_score, dim = -1) # b, head, h*w, h*w
        x = torch.matmul(attation_score, v) # b, head, h*w, h*w @ b, head, h*w, c//head -> b, head, h*w, c//head
        x = x.permute(0, 2, 1, 3).reshape(batch_size, seq, -1) # b, head, h*w, c//head -> b, h*w, head, c//head -> b, h*w, c
        x = self.latent_to_out(x).permute(0,2,1).reshape(*v_shape) #b, h*w, c -> b, c, h*w -> b, c, h, w
        return x



class image_text_multihead_cross_attation_block(nn.Module):
    def __init__(self,q_in_dim, k_in_dim, v_in_dim, latent_dim, out_dim, heads = 8):
        #支持使用图像作为注意力的query，文本作为注意力的key和value
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

    def forward(self, q,k,v):
        # q: img (batch, channel, height, width)
        # kv:text(batch, text_seq, text_emb)
        heads = self.heads
        latent_dim = self.latent_dim
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
        return x


class KL_VAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        img_h = args.input_image_size
        img_in_dim = args.input_image_dims
        acti_func = args.train_activate_func

        self.encoder = nn.Sequential(
                                    #encoder_input_block
                                    conv_block_acti_func_forward(img_in_dim, 128, img_h, acti_func),

                                    #downsample_block
                                    nn.Sequential(residual_block_acti_func_forward(128, 128, int(img_h/1), acti_func),
                                                residual_block_acti_func_forward(128, 256, int(img_h/2), acti_func),conv_downsampling(256, acti_func),
                                                residual_block_acti_func_forward(256, 256, int(img_h/4), acti_func),conv_downsampling(256, acti_func),
                                                residual_block_acti_func_forward(256, 512, int(img_h/8), acti_func),conv_downsampling(512, acti_func)),
                                     
                                     #encoder_middle_block
                                     nn.Sequential(residual_block_acti_func_forward(512, 512, int(img_h/8), acti_func),
                                                  multihead_attation_block(int(img_h/8)**2, int(img_h/8)**2, int(img_h/8)**2, int(img_h/8)**2, int(img_h/8)**2),
                                                  residual_block_acti_func_forward(512, 512, int(img_h/8), acti_func)),
                                     
                                     #encoder_out_block
                                     conv_block_acti_func_forward(512, 8, int(img_h/8), acti_func)
                                     )

        self.decoder = nn.Sequential(
                                    #decoder_input_block
                                    conv_block_acti_func_forward(4, 512, int(img_h/8), acti_func),
                                    #decoder_middle_block
                                    nn.Sequential(residual_block_acti_func_forward(512, 512, int(img_h/8), acti_func),
                                                  multihead_attation_block(int(img_h/8)**2, int(img_h/8)**2, int(img_h/8)**2, int(img_h/8)**2, int(img_h/8)**2),
                                                  residual_block_acti_func_forward(512, 512, int(img_h/8), acti_func)),
                                    #upsample_block
                                    nn.Sequential(residual_block_acti_func_forward(512, 256, int(img_h/8), acti_func), 
                                                residual_block_acti_func_forward(256, 256, int(img_h/4), acti_func), bilinear_upsampling_block(256,acti_func),
                                                residual_block_acti_func_forward(256, 128, int(img_h/2), acti_func), bilinear_upsampling_block(128,acti_func),
                                                residual_block_acti_func_forward(128, 128, int(img_h/1), acti_func), bilinear_upsampling_block(128,acti_func)),
                                    #decoder_output_block
                                    nn.Sequential(conv_block_acti_func_forward(128, img_in_dim, int(img_h/1), acti_func),
                                                  nn.Tanh())
                                    )
        
        self.apply(init_vae_params)

    def forward(self, x):
        # x (batch, channel, height, width), height && width must belong in [-1, 1]
        x = self.encoder(x)
        mu, log_var = x[:, :4], x[:, 4:]
        latent_z = mu + torch.exp(log_var / 2) * torch.randn_like(mu)
        pred_x = self.decoder(latent_z)
        return pred_x, mu, log_var



class KL_VAE2(nn.Module):
    def __init__(self, args):
        #KL_VAE中使用的attation模块存在问题，这个版本使用image_multihead_attation_block
        super().__init__()
        img_h = args.input_image_size
        img_in_dim = args.input_image_dims
        acti_func = args.train_activate_func

        self.encoder = nn.Sequential(
                                    #encoder_input_block
                                    conv_block_acti_func_forward(img_in_dim, 128, img_h, acti_func),

                                    #downsample_block
                                    nn.Sequential(residual_block_acti_func_forward(128, 128, int(img_h/1), acti_func),
                                                residual_block_acti_func_forward(128, 256, int(img_h/2), acti_func),conv_downsampling(256, acti_func),
                                                residual_block_acti_func_forward(256, 256, int(img_h/4), acti_func),conv_downsampling(256, acti_func),
                                                residual_block_acti_func_forward(256, 512, int(img_h/8), acti_func),conv_downsampling(512, acti_func)),
                                     
                                     #encoder_middle_block
                                     nn.Sequential(residual_block_acti_func_forward(512, 512, int(img_h/8), acti_func),
                                                  image_multihead_attation_block(512, 512, 512, 512, 512),
                                                  residual_block_acti_func_forward(512, 512, int(img_h/8), acti_func)),
                                     
                                     #encoder_out_block
                                     conv_block_acti_func_forward(512, 8, int(img_h/8), acti_func)
                                     )

        self.decoder = nn.Sequential(
                                    #decoder_input_block
                                    conv_block_acti_func_forward(4, 512, int(img_h/8), acti_func),
                                    #decoder_middle_block
                                    nn.Sequential(residual_block_acti_func_forward(512, 512, int(img_h/8), acti_func),
                                                  image_multihead_attation_block(512, 512, 512, 512, 512),
                                                  residual_block_acti_func_forward(512, 512, int(img_h/8), acti_func)),
                                    #upsample_block
                                    nn.Sequential(residual_block_acti_func_forward(512, 256, int(img_h/8), acti_func), 
                                                residual_block_acti_func_forward(256, 256, int(img_h/4), acti_func), bilinear_upsampling_block(256, acti_func),
                                                residual_block_acti_func_forward(256, 128, int(img_h/2), acti_func), bilinear_upsampling_block(128, acti_func),
                                                residual_block_acti_func_forward(128, 128, int(img_h/1), acti_func), bilinear_upsampling_block(128, acti_func)),
                                    #decoder_output_block
                                    nn.Sequential(conv_block_acti_func_forward(128, img_in_dim, int(img_h/1), acti_func),
                                                  nn.Tanh())
                                    )
        
        self.apply(init_vae_params)

    def forward(self, x):
        # x (batch, channel, height, width), height && width must belong in [-1, 1]
        x = self.encoder(x)
        mu, log_var = x[:, :4], x[:, 4:]
        latent_z = mu + torch.exp(log_var / 2) * torch.randn_like(mu)
        pred_x = self.decoder(latent_z)
        return pred_x, mu, log_var



class Oriented_KL_VAE2(nn.Module):
    def __init__(self, args):
        #KL_VAE中使用的attation模块存在问题，这个版本使用image_multihead_attation_block
        super().__init__()
        img_h = args.input_image_size
        img_in_dim = args.input_image_dims
        acti_func = args.train_activate_func

        self.encoder = nn.Sequential(
                                    #encoder_input_block
                                    oriented_conv_block(img_in_dim, 128, img_h, acti_func),

                                    #downsample_block
                                    nn.Sequential(oriented_residual_block(128, 128, int(img_h/1), acti_func),
                                                oriented_residual_block(128, 256, int(img_h/2), acti_func),oriented_conv_downsampling(256, acti_func),
                                                oriented_residual_block(256, 256, int(img_h/4), acti_func),oriented_conv_downsampling(256, acti_func),
                                                oriented_residual_block(256, 512, int(img_h/8), acti_func),oriented_conv_downsampling(512, acti_func)),
                                     
                                     #encoder_middle_block
                                     nn.Sequential(oriented_residual_block(512, 512, int(img_h/8), acti_func),
                                                  image_multihead_attation_block(512, 512, 512, 512, 512),
                                                  oriented_residual_block(512, 512, int(img_h/8), acti_func)),
                                     
                                     #encoder_out_block
                                     oriented_conv_block(512, 8, int(img_h/8), acti_func)
                                     )

        self.decoder = nn.Sequential(
                                    #decoder_input_block
                                    oriented_conv_block(4, 512, int(img_h/8), acti_func),
                                    #decoder_middle_block
                                    nn.Sequential(oriented_residual_block(512, 512, int(img_h/8), acti_func),
                                                  image_multihead_attation_block(512, 512, 512, 512, 512),
                                                  oriented_residual_block(512, 512, int(img_h/8), acti_func)),
                                    #upsample_block
                                    nn.Sequential(oriented_residual_block(512, 256, int(img_h/8), acti_func), 
                                                oriented_residual_block(256, 256, int(img_h/4), acti_func), oriented_bilinear_upsampling_block(256, acti_func),
                                                oriented_residual_block(256, 128, int(img_h/2), acti_func), oriented_bilinear_upsampling_block(128, acti_func),
                                                oriented_residual_block(128, 128, int(img_h/1), acti_func), oriented_bilinear_upsampling_block(128, acti_func)),
                                    #decoder_output_block
                                    nn.Sequential(oriented_conv_block(128, img_in_dim, int(img_h/1), acti_func),
                                                  nn.Tanh())
                                    )
        
        self.apply(init_vae_params)

    def forward(self, x):
        # x (batch, channel, height, width), height && width must belong in [-1, 1]
        x = self.encoder(x)
        mu, log_var = x[:, :4], x[:, 4:]
        latent_z = mu + torch.exp(log_var / 2) * torch.randn_like(mu)
        pred_x = self.decoder(latent_z)
        return pred_x, mu, log_var

    
class Unet3(nn.Module):
    def __init__(self, args):
        #增加文本输入
        super().__init__()
        h = args.input_image_size
        c_in = args.input_image_dims
        c_out = args.output_image_dims
        time_dims = args.time_embedding_dims
        activate_func_name = args.train_activate_func
        text_embedding_dims = args.text_embedding_dims
        T = args.time_steps
        
        self.time_embedding = nn.Embedding(T, time_dims)
        
        self.conv1 = conv_block(c_in, 64, int(h), activate_func_name)
        self.time1 = time_embedding_block(time_dims, 64, int(h), activate_func_name)
        self.res1 = residual_block_acti_func_forward(64, 64, int(h), activate_func_name)
        
        self.downsample1 = conv_downsampling(64)
        
        self.conv2 = conv_block(64, 128, int(h/2), activate_func_name)
        self.time2 = time_embedding_block(time_dims, 128, int(h/2), activate_func_name)
        self.res2 = residual_block_acti_func_forward(128, 128, int(h/2), activate_func_name)
        
        self.downsample2 = conv_downsampling(128)
        
        self.conv3 = conv_block(128, 256, int(h/4), activate_func_name)
        self.time3 = time_embedding_block(time_dims, 256, int(h/4), activate_func_name)
        self.res3 = residual_block_acti_func_forward(256, 256, int(h/4), activate_func_name)
        
        self.downsample3 = conv_downsampling(256)
        
        self.conv4 = conv_block(256, 512, int(h/8), activate_func_name)
        self.time4 = time_embedding_block(time_dims, 512, int(h/8), activate_func_name)
        self.res4 = residual_block_acti_func_forward(512, 512, int(h/8), activate_func_name)
        
        self.downsample4 = conv_downsampling(512)
        
        self.bottleneckle_conv = conv_block(512, 1024, int(h/16), activate_func_name)
        self.bottleneckle_time = time_embedding_block(time_dims, 1024, int(h/16), activate_func_name)
        self.bottleneckle_res = residual_block_with_cross_attation(1024, 1024, text_embedding_dims, int(h/16), activate_func_name)
        
        self.upsample5 = upsampling(1024)
        
        self.conv5 = conv_block(1024 + 512, 768, int(h/8), activate_func_name)
        self.time5 = time_embedding_block(time_dims, 768, int(h/8), activate_func_name)
        self.res5 = residual_block_with_cross_attation(768, 768, text_embedding_dims, int(h/8), activate_func_name)
        
        self.upsample6 = upsampling(768)
        
        self.conv6 = conv_block(768 + 256, 512, int(h/4), activate_func_name)
        self.time6 = time_embedding_block(time_dims, 512, int(h/4), activate_func_name)
        self.res6 = residual_block_with_cross_attation(512, 512, text_embedding_dims, int(h/4), activate_func_name)
        
        self.upsample7 = upsampling(512)
        
        self.conv7 = conv_block(512 + 128, 320, int(h/2), activate_func_name)
        self.time7 = time_embedding_block(time_dims, 320, int(h/2), activate_func_name)
        self.res7 = residual_block_with_cross_attation(320, 320, text_embedding_dims, int(h/2), activate_func_name)
        
        self.upsample8 = upsampling(320)
        
        self.conv8 = conv_block(320 + 64, 192, int(h), activate_func_name)
        self.time8 = time_embedding_block(time_dims, 192, int(h), activate_func_name)
        self.res8 = residual_block_with_cross_attation(192, 192, text_embedding_dims, int(h), activate_func_name)
        
        self.output = nn.Sequential(nn.GroupNorm(num_groups=32, num_channels=192),
                                    get_activate_func(activate_func_name),
                                    nn.Conv2d(192, 96, kernel_size=1, stride=1),
                                    nn.GroupNorm(num_groups=32, num_channels=96),
                                    get_activate_func(activate_func_name),
                                    nn.Conv2d(96, 32, kernel_size=1, stride=1),
                                    nn.GroupNorm(num_groups=32, num_channels=32),
                                    nn.Conv2d(32, c_out, kernel_size=1, stride=1))
        
        self.apply(init_ddpm_params)
        
    def forward(self, x, t, text_embedding):
        t = self.time_embedding(t-1)
        t_1 = self.time1(t)
        t_2 = self.time2(t)
        t_3 = self.time3(t)
        t_4 = self.time4(t)
        t_bn = self.bottleneckle_time(t)
        t_5 = self.time5(t)
        t_6 = self.time6(t)
        t_7 = self.time7(t)
        t_8 = self.time8(t)
        
        x = self.conv1(x)
        x = x + t_1
        x_skp1 = self.res1(x)
        x = self.downsample1(x_skp1)
        
        x = self.conv2(x)
        x = x + t_2
        x_skp2 = self.res2(x)
        x = self.downsample2(x_skp2)
        
        x = self.conv3(x)
        x = x + t_3
        x_skp3 = self.res3(x)
        x = self.downsample3(x_skp3)
        
        x = self.conv4(x)
        x = x + t_4
        x_skp4 = self.res4(x)
        x = self.downsample3(x_skp4)
        
        x = self.bottleneckle_conv(x)
        x = x + t_bn
        x = self.bottleneckle_res(x)
        
        x = self.upsample5(x)
        x = torch.cat((x, x_skp4), dim=1)
        x = self.conv5(x)
        x = x + t_5
        x = self.res5(x)
        
        x = self.upsample6(x)
        x = torch.cat((x, x_skp3), dim=1)
        x = self.conv6(x)
        x = x + t_6
        x = self.res6(x)
        
        x = self.upsample7(x)
        x = torch.cat((x, x_skp2), dim=1)
        x = self.conv7(x)
        x = x + t_7
        x = self.res7(x)
        
        x = self.upsample8(x)
        x = torch.cat((x, x_skp1), dim=1)
        x = self.conv8(x)
        x = x + t_8
        x = self.res8(x)
        
        x = self.output(x)
        
        return x


