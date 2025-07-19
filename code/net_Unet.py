import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.cm as cm
from encoder.net1d import Net1D
from encoder.resnet18_pyramid import ResNet18_pyramidal
from vision.unet_large import UNet_large
from loss import (
    masked_region_MSE_Loss,
    entire_region_MSE_Loss,
    SSIMLoss,
    DepthGANLoss,
)
from metric import compute_psnr, compute_lpips, SSIM, RelativeError


class Acoustic_Encoder(nn.Module):
    def __init__(self):
        super(Acoustic_Encoder, self).__init__()
        self.RIR_1D = True
        if self.RIR_1D:  # 1D input
            self.model = Net1D(  # check Net1d.py line 464
                in_channels=4,
                base_filters=64,
                ratio=1.0,
                filter_list=[64, 160, 160, 400, 400, 1024, 1024],
                m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
                kernel_size=16,
                stride=2,
                groups_width=16,
                verbose=False,
                n_classes=256,
            )
        else:  # 2D input
            self.model = ResNet18_pyramidal(
                dropout_prob=0.0,
                in_channel=8,
                out_dim=128,
                out_size=(32, 64),
                out_channel=256,
                spatial_feature=True,
            )

    def forward(self, batch_rir):
        encoded_rir = self.model(batch_rir)
        return encoded_rir  # [BS x 256] or [BS x 256 x 32 x 64]


class GlobalwisePixelEstimate(nn.Module):
    def __init__(self, input_channel=256):
        super(GlobalwisePixelEstimate, self).__init__()
        self.global_conv = nn.Conv2d(
            in_channels=16,
            out_channels=16 * 16 * 32,  # 2048 filters
            kernel_size=(32, 64),
            stride=1,
            padding=0,
        )
        self.channel_inc = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.global_conv(x)
        x = x.view(x.shape[0], 16, 16, 32)
        x = self.channel_inc(x)
        x = self.relu(x)
        return x


class FiLM(nn.Module):
    def __init__(self, conditioning_dim, num_channels):
        super(FiLM, self).__init__()
        self.fc = nn.Linear(conditioning_dim, num_channels * 2)

    def forward(self, features, condition):
        gamma_beta = self.fc(condition)  # (B, 2*num_channels)
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # 각 (B, num_channels)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, num_channels, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * features + beta


class DepthExtract(nn.Module):
    def __init__(self):
        super(DepthExtract, self).__init__()
        self.midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
        for param in self.midas_model.parameters():
            param.requires_grad = False
        self.midas_model.eval()

    def forward(self, masked_img):
        masekd_depth = self.midas_model(masked_img)  # [B x 3 x H x W]
        masekd_depth = torch.nn.functional.interpolate(
            masekd_depth.unsqueeze(1),  # [B x 1 x H x W]
            size=((256, 512)),  # [B x 1 x 256 x 512]
            mode="bilinear",
            align_corners=False,
        )
        return masekd_depth


class UnetModel(nn.Module):
    def __init__(self):
        super(UnetModel, self).__init__()

        self.acoustic_encoder = Acoustic_Encoder()
        self.globalconv = GlobalwisePixelEstimate()
        self.Depth = DepthExtract()
        self.UNet_large = UNet_large()

        self.Film_l1 = FiLM(2, 32)
        self.Film_l2 = FiLM(2, 64)
        self.Film_l3 = FiLM(2, 128)
        self.Film_l4 = FiLM(2, 256)

        self.channel_inc_l1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.channel_inc_l2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.channel_inc_l3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.channel_inc_l4 = nn.Conv2d(64, 256, kernel_size=1, stride=1)
        self.relu = nn.ReLU()

        self.latent_size_pooling = nn.AdaptiveAvgPool2d((16, 32))

        self.context_loss = entire_region_MSE_Loss()
        self.recon_loss = masked_region_MSE_Loss()
        self.structural_loss = SSIMLoss()
        self.latent_loss = entire_region_MSE_Loss()
        self.DepthGAN_loss = DepthGANLoss()

        self.metric_psnr = compute_psnr()
        self.metric_lpipis = compute_lpips()
        self.metric_ssim = SSIM()
        self.metric_rel = RelativeError()

        self.loss_scale = 2 * 10e-2

        self.context_loss_scale = 0.3 * 10e-4
        self.structural_loss_scale = 0.2 * 10e2
        self.recon_loss_scale = 1.0 * 10e-3
        self.latent_loss_scale = 0.02 * 10e-4
        self.DepthGAN_loss_scale = 2 * 10e0

        self.monitoring = True

    def find_masked_start_pos(self, masked_img):
        batch_start_positions = []
        B = masked_img.shape[0]
        W = masked_img.shape[3]
        for i in range(B):
            found = False
            for start_pos in range(0, W - 256 + 1):
                masked_region = masked_img[i, :, :, start_pos]
                if torch.all(masked_region == 0):
                    # print(f"batch {i} start_position is {start_pos}")
                    batch_start_positions.append(start_pos)
                    found = True
                    break
            if not found:
                print(f"batch {i} can not find masked region")
                batch_start_positions.append(-1)

        return torch.tensor(batch_start_positions)

    def save_depth_as_image(self, depth_map, save_path, h, w):
        if depth_map.ndimension() == 3 and depth_map.shape[0] > 1:
            depth_map = depth_map[0]

            # Imaging
        depth_array = depth_map.squeeze().detach().cpu().numpy()
        depth_array = depth_array.reshape((h, w))

        depth_normalized = (
            255
            * (depth_array - np.min(depth_array))
            / (np.max(depth_array) - np.min(depth_array))
        ).astype(np.uint8)
        depth_image = Image.fromarray(depth_normalized)
        depth_image.save(save_path)

    def gaussian_blur(self, tensor, kernel_size=5, sigma=1.0):
        # 1차원 Gaussian 커널 생성
        def gauss_kernel(size, sigma):
            x = torch.arange(-size // 2 + 1.0, size // 2 + 1.0)
            kernel = torch.exp(-(x**2) / (2 * sigma**2))
            kernel = kernel / kernel.sum()
            return kernel

        kernel_1d = gauss_kernel(kernel_size, sigma).to(tensor.device)
        # 2D Gaussian 커널 생성
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d.expand(tensor.shape[1], 1, kernel_size, kernel_size)

        padding = kernel_size // 2
        blurred = nn.functional.conv2d(
            tensor, kernel_2d, padding=padding, groups=tensor.shape[1]
        )
        return blurred

    def save_depth_with_color_border(self, depth_tensor, save_path, h, w, pos_out, region_width=128):
        depth_array = depth_tensor.squeeze().detach().cpu().numpy().reshape((h, w))
        depth_normalized = (
            255 * (depth_array - np.min(depth_array)) / (np.max(depth_array) - np.min(depth_array))
        ).astype(np.uint8)

        colormap = cm.get_cmap('jet_r')
        depth_color = colormap(depth_normalized / 255.0)  # shape: (h, w, 4)
        
        depth_color = (depth_color[:, :, :3] * 255).astype(np.uint8)
    
        depth_image = Image.fromarray(depth_color)

        draw = ImageDraw.Draw(depth_image)
        draw.rectangle(
            [pos_out, 0, pos_out + region_width, h],
            outline=(0, 0, 0),
            width=3
        )
        depth_image.save(save_path)

    def apply_mask_attention(
        self,
        latent,
        mask_positions,
        region_width,
        latent_scale_factor,
        high_weight=1.0,
        low_weight=0.01,
    ):

        B, C, H, W = latent.shape
        attn_maps = []
        for i in range(B):
            pos = mask_positions[i].item()
            pos_latent = int(pos // latent_scale_factor)
            region_width_latent = int(region_width // latent_scale_factor)
            attn = torch.ones(H, W, device=latent.device) * low_weight
            attn[:, pos_latent : min(pos_latent + region_width_latent, W)] = high_weight
            attn_maps.append(attn)
        attn_maps = torch.stack(attn_maps, dim=0)  # [B, H, W]
        attn_maps = attn_maps.unsqueeze(1)  # [B, 1, H, W]
        return latent * attn_maps

    def forward(self, masked_img, GT_img, rir, GT_depth, mic_info, exp_order, metric):

        encoded_rir = self.acoustic_encoder(rir)  # [B x 16 x 32 x 64]
        latent_feature = self.globalconv(encoded_rir) 

        l4 = self.Film_l4(self.channel_inc_l4(latent_feature), mic_info)  # [B x 256 x 16 x 32]
        l3 = self.Film_l3(self.channel_inc_l3(l4), mic_info)  # [B x 128 x 32 x 64]
        l2 = self.Film_l2(self.channel_inc_l2(l3), mic_info)  # [B x 64 x 64 x 128]
        l1 = self.Film_l1(self.channel_inc_l1(l2), mic_info)  # [B x 32 x 128 x 256]
 
        l4 = self.relu(l4)
        l3 = self.relu(l3)
        l2 = self.relu(l2)
        l1 = self.relu(l1)

        mask_pos = self.find_masked_start_pos(masked_img)

        l4 = self.apply_mask_attention(
            l4,
            mask_pos,
            region_width=256,
            latent_scale_factor=32,
            high_weight=1.0,
            low_weight=0.3,
        )
        l3 = self.apply_mask_attention(
            l3,
            mask_pos,
            region_width=256,
            latent_scale_factor=16,
            high_weight=1.0,
            low_weight=0.2,
        )
        l2 = self.apply_mask_attention(
            l2,
            mask_pos,
            region_width=256,
            latent_scale_factor=8,
            high_weight=1.0,
            low_weight=0.1,
        )
        l1 = self.apply_mask_attention(
            l1,
            mask_pos,
            region_width=256,
            latent_scale_factor=4,
            high_weight=1.0,
            low_weight=0.01,
        )


        masked_depth = self.Depth(masked_img)
        gen_depth = self.UNet_large(masked_depth, l4, l3, l2, l1) 
        # gen_depth = self.gaussian_blur(gen_depth, kernel_size=7, sigma=1.0)
        GT_depth = GT_depth.unsqueeze(1)     # (B, 1, 256, 512)
        
        
        if metric:
            psnr = self.metric_psnr(GT_depth, gen_depth, (mask_pos // 2), (256 // 2))
            lpips = self.metric_lpipis(GT_depth, gen_depth, (mask_pos // 2), (256 // 2))
            ssim = self.metric_ssim(GT_depth, gen_depth, (mask_pos // 2), (256 // 2))
            rel = self.metric_rel(GT_depth, gen_depth, (mask_pos // 2), (256 // 2))
            # print(
            #     f"PSNR: {psnr.item():.4f}, "
            #     f"LPIPS: {lpips.item():.4f}, "
            #     f"SSIM: {ssim.item():.4f}, "
            #     f"Rel: {rel.item():.4f}"
            # )
            return psnr, lpips, ssim, rel

        context_loss = self.context_loss_scale * self.context_loss(GT_depth, gen_depth)
        recon_loss = self.recon_loss_scale * self.recon_loss(
            GT_depth, gen_depth, (mask_pos // 2), (256 // 2)
        )
        structural_loss = self.structural_loss_scale * self.structural_loss(
            GT_depth, gen_depth, (mask_pos // 2), (256 // 2)
        )
        latent_loss = self.latent_loss_scale * self.latent_loss(
            self.latent_size_pooling(GT_depth),
            latent_feature.mean(dim=1, keepdim=True),
        )
        DepthGAN_loss = self.DepthGAN_loss_scale * self.DepthGAN_loss(
            GT_depth, gen_depth, (mask_pos // 2), (256 // 2)
        )



        loss = self.loss_scale * (
            context_loss
            + recon_loss
            + structural_loss
            + latent_loss
            + DepthGAN_loss
        )

        # print(
        #     f"Context: {context_loss}, "
        #     f"Recon: {recon_loss}, "
        #     f"Structural: {structural_loss}, "
        #     f"latent: {latent_loss}, "
        #     f"Depth GAN: {DepthGAN_loss}, "
        #     f"total loss: {loss}"
        # )

        if self.monitoring:  # Monitoring
            monitor_path = "monitoring/March/ablation_large"
            os.makedirs(monitor_path, exist_ok=True)
            self.save_depth_with_color_border(
                gen_depth[0],
                f"{monitor_path}/GEN_{exp_order}.png",
                256,
                512,
                mask_pos // 2,
                region_width=128,
            )
            self.save_depth_with_color_border(
                GT_depth[0],
                f"{monitor_path}/GT_{exp_order}.png",
                256,
                512,
                mask_pos // 2,
                region_width=128,
            )
            latent_tensor = latent_feature.mean(dim=1, keepdim=True)  # batch mean
            self.save_depth_as_image(
                latent_tensor[0],
                f"{monitor_path}/Latent_{exp_order}.png",
                16,
                32,
            )

        return loss
