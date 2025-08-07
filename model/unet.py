from model.model_block import *

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
        t = self.time_embedding(t)
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
        x = self.downsample4(x_skp4)
        
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



class Unet_With_Text_Condition(nn.Module):
    def __init__(self, args):
        #增加文本输入
        super().__init__()
        h = args.input_image_size
        c_in = args.input_image_dims
        c_out = args.output_image_dims
        time_dims = args.time_embedding_dims
        activate_func_name = args.train_activate_func
        args.text_embedding_dims = 512 if args.embedding_model == "clip" else None
        text_embedding_dims = args.text_embedding_dims
        T = args.time_steps
        
        self.time_embedding = nn.Embedding(T, time_dims)
        
        self.conv1 = conv_first_block(c_in, 64, int(h), activate_func_name)
        self.time1 = time_embedding_block(time_dims, 64, int(h), activate_func_name)
        self.res1 = residual_block_acti_func_forward(64, 64, int(h), activate_func_name)
        
        self.downsample1 = conv_downsampling(64, activate_func_name)
        
        self.conv2 = conv_first_block(64, 128, int(h/2), activate_func_name)
        self.time2 = time_embedding_block(time_dims, 128, int(h/2), activate_func_name)
        self.res2 = residual_block_acti_func_forward(128, 128, int(h/2), activate_func_name)
        
        self.downsample2 = conv_downsampling(128, activate_func_name)
        
        self.conv3 = conv_first_block(128, 256, int(h/4), activate_func_name)
        self.time3 = time_embedding_block(time_dims, 256, int(h/4), activate_func_name)
        self.res3 = residual_block_acti_func_forward(256, 256, int(h/4), activate_func_name)
        
        self.downsample3 = conv_downsampling(256, activate_func_name)
        
        self.conv4 = conv_first_block(256, 512, int(h/8), activate_func_name)
        self.time4 = time_embedding_block(time_dims, 512, int(h/8), activate_func_name)
        self.res4 = residual_block_acti_func_forward(512, 512, int(h/8), activate_func_name)
        
        self.downsample4 = conv_downsampling(512, activate_func_name)
        
        self.bottleneckle_conv = conv_first_block(512, 1024, int(h/16), activate_func_name)
        self.bottleneckle_time = time_embedding_block(time_dims, 1024, int(h/16), activate_func_name)
        self.bottleneckle_res = residual_block_with_cross_attation(1024, 1024, text_embedding_dims, int(h/16), activate_func_name)
        
        self.upsample5 = bilinear_upsampling_block(1024, activate_func_name)
        
        self.conv5 = conv_first_block(1024 + 512, 768, int(h/8), activate_func_name)
        self.time5 = time_embedding_block(time_dims, 768, int(h/8), activate_func_name)
        self.res5 = residual_block_with_cross_attation(768, 768, text_embedding_dims, int(h/8), activate_func_name)
        
        self.upsample6 = bilinear_upsampling_block(768, activate_func_name)
        
        self.conv6 = conv_first_block(768 + 256, 512, int(h/4), activate_func_name)
        self.time6 = time_embedding_block(time_dims, 512, int(h/4), activate_func_name)
        self.res6 = residual_block_with_cross_attation(512, 512, text_embedding_dims, int(h/4), activate_func_name)
        
        self.upsample7 = bilinear_upsampling_block(512, activate_func_name)
        
        self.conv7 = conv_first_block(512 + 128, 320, int(h/2), activate_func_name)
        self.time7 = time_embedding_block(time_dims, 320, int(h/2), activate_func_name)
        self.res7 = residual_block_with_cross_attation(320, 320, text_embedding_dims, int(h/2), activate_func_name)
        
        self.upsample8 = bilinear_upsampling_block(320, activate_func_name)
        
        self.conv8 = conv_first_block(320 + 64, 192, int(h), activate_func_name)
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
        t = self.time_embedding(t)
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
        x = self.downsample4(x_skp4)
        
        x = self.bottleneckle_conv(x)
        x = x + t_bn
        x = self.bottleneckle_res(x, text_embedding)
        
        x = self.upsample5(x)
        x = torch.cat((x, x_skp4), dim=1)
        x = self.conv5(x)
        x = x + t_5
        x = self.res5(x, text_embedding)
        
        x = self.upsample6(x)
        x = torch.cat((x, x_skp3), dim=1)
        x = self.conv6(x)
        x = x + t_6
        x = self.res6(x, text_embedding)
        
        x = self.upsample7(x)
        x = torch.cat((x, x_skp2), dim=1)
        x = self.conv7(x)
        x = x + t_7
        x = self.res7(x, text_embedding)
        
        x = self.upsample8(x)
        x = torch.cat((x, x_skp1), dim=1)
        x = self.conv8(x)
        x = x + t_8
        x = self.res8(x, text_embedding)
        
        x = self.output(x)
        
        return x


class Unet_Without_Condition(nn.Module):
    def __init__(self, args):
        #增加文本输入
        super().__init__()
        h = args.input_image_size
        c_in = args.input_image_dims
        c_out = args.output_image_dims
        time_dims = args.time_embedding_dims
        activate_func_name = args.train_activate_func
        T = args.time_steps
        
        self.time_embedding = nn.Embedding(T, time_dims)
        
        self.conv1 = conv_first_block(c_in, 64, int(h), activate_func_name)
        self.time1 = time_embedding_block(time_dims, 64, int(h), activate_func_name)
        # self.res1 = residual_block_acti_func_forward(64, 64, int(h), activate_func_name)
        self.res1 = residual_block(64, 64, int(h), activate_func_name)
        
        self.downsample1 = downsampling() # conv_downsampling(64, activate_func_name)
        
        self.conv2 = conv_first_block(64, 128, int(h/2), activate_func_name)
        self.time2 = time_embedding_block(time_dims, 128, int(h/2), activate_func_name)
        # self.res2 = residual_block_acti_func_forward(128, 128, int(h/2), activate_func_name)
        self.res2 = residual_block(128, 128, int(h/2), activate_func_name)
        
        self.downsample2 = downsampling() # conv_downsampling(128, activate_func_name)
        
        self.conv3 = conv_first_block(128, 256, int(h/4), activate_func_name)
        self.time3 = time_embedding_block(time_dims, 256, int(h/4), activate_func_name)
        # self.res3 = residual_block_acti_func_forward(256, 256, int(h/4), activate_func_name)
        self.res3 = residual_block(256, 256, int(h/4), activate_func_name)
        
        self.downsample3 = downsampling() # conv_downsampling(256, activate_func_name)
        
        self.conv4 = conv_first_block(256, 512, int(h/8), activate_func_name)
        self.time4 = time_embedding_block(time_dims, 512, int(h/8), activate_func_name)
        # self.res4 = residual_block_acti_func_forward(512, 512, int(h/8), activate_func_name)
        self.res4 = residual_block(512, 512, int(h/8), activate_func_name)
        
        self.downsample4 = downsampling() # conv_downsampling(512, activate_func_name)
        
        self.bottleneckle_conv = conv_first_block(512, 1024, int(h/16), activate_func_name)
        self.bottleneckle_time = time_embedding_block(time_dims, 1024, int(h/16), activate_func_name)
        # self.bottleneckle_res = residual_block_acti_func_forward(1024, 1024, int(h/16), activate_func_name)
        self.bottleneckle_res = residual_block(1024, 1024, int(h/16), activate_func_name)
        
        self.upsample5 = upsampling(1024) # bilinear_upsampling_block(1024, activate_func_name)
        
        self.conv5 = conv_first_block(1024 + 512, 768, int(h/8), activate_func_name)
        self.time5 = time_embedding_block(time_dims, 768, int(h/8), activate_func_name)
        # self.res5 = residual_block_acti_func_forward(768, 768, int(h/8), activate_func_name)
        self.res5 = residual_block(768, 768, int(h/8), activate_func_name)
        
        self.upsample6 = upsampling(768) # bilinear_upsampling_block(768, activate_func_name)
        
        self.conv6 = conv_first_block(768 + 256, 512, int(h/4), activate_func_name)
        self.time6 = time_embedding_block(time_dims, 512, int(h/4), activate_func_name)
        # self.res6 = residual_block_acti_func_forward(512, 512, int(h/4), activate_func_name)
        self.res6 = residual_block(512, 512, int(h/4), activate_func_name)
        
        self.upsample7 = upsampling(512) # bilinear_upsampling_block(512, activate_func_name)
        
        self.conv7 = conv_first_block(512 + 128, 320, int(h/2), activate_func_name)
        self.time7 = time_embedding_block(time_dims, 320, int(h/2), activate_func_name)
        # self.res7 = residual_block_acti_func_forward(320, 320, int(h/2), activate_func_name)
        self.res7 = residual_block(320, 320, int(h/2), activate_func_name)
        
        self.upsample8 = upsampling(320) # bilinear_upsampling_block(320, activate_func_name)
        
        self.conv8 = conv_first_block(320 + 64, 192, int(h), activate_func_name)
        self.time8 = time_embedding_block(time_dims, 192, int(h), activate_func_name)
        # self.res8 = residual_block_acti_func_forward(192, 192, int(h), activate_func_name)
        self.res8 = residual_block(192, 192, int(h), activate_func_name)
        
        self.output = nn.Sequential(
                                    nn.GroupNorm(num_groups=32, num_channels=192),
                                    nn.Conv2d(192, 96, kernel_size=1, stride=1),
                                    get_activate_func(activate_func_name),
                                    nn.GroupNorm(num_groups=32, num_channels=96),
                                    nn.Conv2d(96, 32, kernel_size=1, stride=1),
                                    get_activate_func(activate_func_name),
                                    # nn.GroupNorm(num_groups=32, num_channels=32),
                                    nn.Conv2d(32, c_out, kernel_size=1, stride=1)
                                    )


        # #输出层使用归一化层可能会导致偏红色和黄色，因为归一化层会弱化蓝色通道，尝试下面的方法
        # #这个也不好，训练之后的图片整体都是蓝色的
        # self.output = nn.Sequential(
        #                             nn.Conv2d(192, 96, kernel_size=1, stride=1),
        #                             get_activate_func(activate_func_name),
        #                             nn.Conv2d(96, 32, kernel_size=1, stride=1),
        #                             get_activate_func(activate_func_name),
        #                             nn.Conv2d(32, c_out, kernel_size=1, stride=1)
        #                             )

        # #尝试这种最简单的，也不好，整体偏红
        # self.output = nn.Sequential(
        #                             nn.GroupNorm(num_groups=32, num_channels=192),
        #                             get_activate_func(activate_func_name),
        #                             nn.Conv2d(192, c_out, kernel_size=3, stride=1, padding = 1)
        #                             )
        
        self.apply(init_ddpm_params)
        
    def forward(self, x, t):
        t = self.time_embedding(t)
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
        x = self.downsample4(x_skp4)
        
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




class Unet3(nn.Module):
    def __init__(self, args):
        #在Unet2的基础上进行升级，增大参数量
        #1、将时间embedding的嵌入位置调整到残差模块的两个卷积之间
        #2、将所有残差块在残差连接之前的最后一个输出的初始权重设置为0
        #3、将上采样的维度数量变化，修改为 2*channel -> channel 的模式
        #4、每个层级内按如下顺序组织：2个残差模块，1个自注意力模块，1个残差模块
        super().__init__()
        h = args.input_image_size
        c_in = args.input_image_dims
        c_out = args.output_image_dims
        time_dims = args.time_embedding_dims
        activate_func_name = args.train_activate_func
        # args.text_embedding_dims = 512 if args.embedding_model == "clip" else None
        # text_embedding_dims = args.text_embedding_dims
        T = args.time_steps
        
        self.time_embedding = nn.Embedding(T, time_dims)
        self.input = nn.Conv2d(c_in, 256, kernel_size=3, stride=1, padding=1)
        down_block_list = [Unet3_res_blocks(256, 256, time_dims, activate_func_name, int(h), downsample = True),
                           Unet3_res_blocks(256, 512, time_dims, activate_func_name, int(h/2), downsample = True, use_attation = True),
                           Unet3_res_blocks(512, 1024, time_dims, activate_func_name, int(h/4), downsample = True, use_attation = True),
                           Unet3_res_blocks(1024, 1024, time_dims, activate_func_name, int(h/8), downsample = True, use_attation = True)
        ]
        self.down_blocks = nn.ModuleList(down_block_list)

        self.bottleneckle = Unet3_res_blocks(1024, 1024, time_dims, activate_func_name, int(h/16), use_attation = True)
        
        up_block_list = [Unet3_res_blocks(1024*2, 1024, time_dims, activate_func_name, int(h/16), upsample = True, use_attation = True),
                        Unet3_res_blocks(1024*2, 512, time_dims, activate_func_name, int(h/8), upsample = True, use_attation = True),
                        Unet3_res_blocks(512*2, 256, time_dims, activate_func_name, int(h/4), upsample = True, use_attation = True),
                        Unet3_res_blocks(256*2, 256, time_dims, activate_func_name, int(h/2), upsample = True)
        ]
        self.up_blocks = nn.ModuleList(up_block_list)

        self.output = nn.Sequential(nn.GroupNorm(num_groups=32, num_channels=256),
                                    nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                                    get_activate_func(activate_func_name),
                                    nn.GroupNorm(num_groups=32, num_channels=128),
                                    nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                                    get_activate_func(activate_func_name),
                                    nn.Conv2d(64, c_out, kernel_size=3, stride=1, padding=1))
        
        self.apply(init_ddpm_params)
        
    def forward(self, x, t):
        time_embedding = self.time_embedding(t)
        x = self.input(x)

        #downsample
        res_list = []
        for down_module in self.down_blocks:
            x, res = down_module(x, time_embedding)
            res_list.append(res)

        #bottleneck
        x, _ = self.bottleneckle(x, time_embedding)

        #upsample
        assert len(self.up_blocks) == len(res_list), "错误，Unet的下采样和上采样次数不一致，无法进行拼接，请检查"
        for up_module in self.up_blocks:
            res = res_list.pop()
            x, _ = up_module(x, time_embedding, res)

        x = self.output(x)
        return x
