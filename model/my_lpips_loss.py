
from torchvision.models import vgg16
import torch
import torch.nn as nn
from checkpoint.lpips_weights import lpips_offical_weights

class LPIPS(nn.Module):
    def __init__(self, rank, model_name, lpips_weights = lpips_offical_weights, yolox_weight_path = "./checkpoint/yolox_backbone.pth"):
        super().__init__()
        self.model_name = model_name
        if model_name == "vgg16":
            self.n_layers = 30
            self.rank = rank
            self.channel_weights_list = lpips_weights
            if rank == 0:
                vgg_model = vgg16(pretrained=True).features[:self.n_layers]
                state_dict = vgg_model.state_dict()
                object_list = [state_dict]
            else:
                vgg_model = vgg16(pretrained=False).features[:self.n_layers]
                state_dict = None
                object_list = [state_dict]

            torch.distributed.broadcast_object_list(object_list, src=0)
            torch.distributed.barrier()

            if rank != 0:
                state_dict = object_list[0]
                vgg_model.load_state_dict(state_dict)

            vgg_model.eval()
            for param in vgg_model.parameters():
                param.requires_grad_(False)
            
            children_model_list = list(vgg_model.children())
            self.conv1 = nn.Sequential(*children_model_list[:4]).to(rank)
            self.conv2 = nn.Sequential(*children_model_list[4:9]).to(rank)
            self.conv3 = nn.Sequential(*children_model_list[9:16]).to(rank)
            self.conv4 = nn.Sequential(*children_model_list[16:23]).to(rank)
            self.conv5 = nn.Sequential(*children_model_list[23:30]).to(rank)
            
            assert len(self.channel_weights_list) == 5, "channel_weights_list的长度必须为5，以匹配5个特征层"
            self.L = len(self.channel_weights_list)
            
            for i in range(len(self.channel_weights_list)):
                self.channel_weights_list[i].requires_grad = False
                self.channel_weights_list[i] = self.channel_weights_list[i].to(rank)
        elif model_name == "yolox":
            from model.yolox import CSPDarknet
            self.yolox_model = CSPDarknet()
            if rank == 0:
                yolox_backbone_statedict = self.abstract_yolox_backbone_statedict(yolox_weight_path)
                yolox_backbone_statedict_list = [yolox_backbone_statedict]
            else:
                yolox_backbone_statedict = None
                yolox_backbone_statedict_list = [yolox_backbone_statedict]
            torch.distributed.broadcast_object_list(yolox_backbone_statedict_list, src=0)
            torch.distributed.barrier()
            self.yolox_model.load_state_dict(yolox_backbone_statedict_list[0], strict=True)
            self.yolox_model.eval()
            for param in self.yolox_model.parameters():
                param.requires_grad_(False)
            self.yolox_model = self.yolox_model.to(rank)
        else:
            raise ValueError("model_name must be vgg16 or yolox, but got {}".format(model_name))

    def abstract_yolox_backbone_statedict(self, pre_trained_model_path):
        pre_trained_state_dict = torch.load(pre_trained_model_path, weights_only=False)
        pre_str = "backbone.backbone."
        new_state_dict = {k.split(pre_str)[1]:v for k, v in pre_trained_state_dict['model'].items() if k.startswith(pre_str)}
        return new_state_dict
    
    def vae_input_image_to_yolox_input_image(self, vae_input_image):
        device = vae_input_image.device
        yolox_input_image = ((vae_input_image + 1) * 0.5 * 255).to(device)
        return yolox_input_image

    def vae_input_image_to_vgg_input_image(self, vae_input_image):
        device = vae_input_image.device
        vgg_input_image = (vae_input_image + 1) * 0.5
        vgg_input_image = (vgg_input_image - torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device))/torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)
        return vgg_input_image

    def safe_L2_norm(self, input_tensor, epsilon = 1e-10):
        scalar = torch.sqrt(torch.sum(input_tensor ** 2, dim=1, keepdim = True) + epsilon)
        return input_tensor/scalar

    def forward(self, input_image, pred_image):
        if self.model_name == "vgg16":
            #将图像像素值从[-1,1]转换成满足均值0方差1的正态分布，满足vgg的图像输入要求
            vgg_input_image = self.vae_input_image_to_vgg_input_image(input_image)
            vgg_pred_image = self.vae_input_image_to_vgg_input_image(pred_image)
            #获取原图和重建图的多尺度特征图
            input_feature1 = self.conv1(vgg_input_image)
            input_feature2 = self.conv2(input_feature1)
            input_feature3 = self.conv3(input_feature2)
            input_feature4 = self.conv4(input_feature3)
            input_feature5 = self.conv5(input_feature4)
            input_image_feature_list = [input_feature1, input_feature2, input_feature3, input_feature4, input_feature5]
            pred_feature1 = self.conv1(vgg_pred_image)
            pred_feature2 = self.conv2(pred_feature1)
            pred_feature3 = self.conv3(pred_feature2)
            pred_feature4 = self.conv4(pred_feature3)
            pred_feature5 = self.conv5(pred_feature4)
            pred_image_feature_list = [pred_feature1, pred_feature2, pred_feature3, pred_feature4, pred_feature5]
            #对所有特征在通道维度进行L2归一化
            input_image_feature_l2_norm_list, pred_image_feature_l2_norm_list = [], []
            for input_feature, pred_feature in zip(input_image_feature_list, pred_image_feature_list):
                input_image_feature_l2_norm_list.append(self.safe_L2_norm(input_feature))
                pred_image_feature_l2_norm_list.append(self.safe_L2_norm(pred_feature))
            #计算输入和预测特征图的L2差别
            diff_list = []
            for input_feature, pred_feature in zip(input_image_feature_l2_norm_list, pred_image_feature_l2_norm_list):
                diff = input_feature - pred_feature
                diff_list.append(diff**2)
            #根据不同层的权重，计算最终的感知损失
            loss = 0.0
            for i in range(self.L):
                loss += torch.sum(self.channel_weights_list[i].view(1,-1,1,1) * diff_list[i], dim = 1, keepdim=True).mean(dim=(1,2,3))
        elif self.model_name == "yolox":
            yolox_input_image = self.vae_input_image_to_yolox_input_image(input_image)
            yolox_pred_image = self.vae_input_image_to_yolox_input_image(pred_image)
            input_image_feature_list = list(self.yolox_model(yolox_input_image).values())
            pred_image_feature_list = list(self.yolox_model(yolox_pred_image).values())
            #对所有特征在通道维度进行L2归一化
            input_image_feature_l2_norm_list, pred_image_feature_l2_norm_list = [], []
            for input_feature, pred_feature in zip(input_image_feature_list, pred_image_feature_list):
                input_image_feature_l2_norm_list.append(self.safe_L2_norm(input_feature))
                pred_image_feature_l2_norm_list.append(self.safe_L2_norm(pred_feature))
            #计算输入和预测特征图的L2差别
            diff_list = []
            for input_feature, pred_feature in zip(input_image_feature_l2_norm_list, pred_image_feature_l2_norm_list):
                diff = input_feature - pred_feature
                diff_list.append(diff**2)
            #这里直接使用不同通道的特征差异作为权重损失，去除尺寸对特征的影响，保留通道数量对特征的影响
            # loss = torch.sum(torch.cat([diff.mean(dim=(2,3)) for diff in diff_list], dim=1), dim=1, keepdim=True)
            loss = torch.sum(torch.cat([diff.mean(dim=(1,2,3), keepdim=True) for diff in diff_list], dim=1), dim=1, keepdim=True)
        else:
            raise ValueError("model_name must be vgg16 or yolox, but got {}".format(self.model_name))
        return loss.mean()
