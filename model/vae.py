
from model.model_block import *

class KL_VAE(nn.Module):
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

