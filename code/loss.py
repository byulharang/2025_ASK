# Class:
#   LPIPS
#   SSIM
#   GAN
#   masked_depth_mse
#   entire_depth_mse

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights
import torch.nn.functional as F
import numpy as np
import torch.nn.utils as utils


class LPIPS(nn.Module):  # input: (B, 3, H, W)
    def __init__(self):
        super(LPIPS, self).__init__()

        vgg_pretrained = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*vgg_pretrained[:4])
        self.slice2 = nn.Sequential(*vgg_pretrained[4:9])
        self.slice3 = nn.Sequential(*vgg_pretrained[9:16])
        self.slice4 = nn.Sequential(*vgg_pretrained[16:23])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, img1, img2, mask_pos, mask_width):
        feature_x = []
        feature_y = []
        masked_img1 = img1[:, :, :, mask_pos : mask_pos + mask_width]
        masked_img2 = img2[:, :, :, mask_pos : mask_pos + mask_width]

        h_x = masked_img1
        h_y = masked_img2

        for slice_module in [self.slice1, self.slice2, self.slice3, self.slice4]:
            h_x = slice_module(h_x)
            h_y = slice_module(h_y)
            feature_x.append(h_x)
            feature_y.append(h_y)

        loss = 0
        for fx, fy in zip(feature_x, feature_y):
            loss += F.mse_loss(fx, fy)
        return loss

    # 사용 예시:
    # lpips = LPIPS()
    # img1, img2: (N, 3, H, W) 텐서, [-1, 1] 혹은 [0,1] 범위로 정규화
    # lpips_score = lpips(img1, img2)


class SSIMLoss(nn.Module):  # input: (B, C, H, W)

    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor(
            [
                np.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
                for x in range(window_size)
            ]
        )
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, sigma=1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def ssim(self, img1, img2, window_size, size_average=True, val_range=None):
        if val_range is None:
            max_val = 1.0 if img1.max() <= 1 else 255
            L = max_val
        else:
            L = val_range

        pad = window_size // 2
        (_, channel, _, _) = img1.size()
        window = self.create_window(window_size, channel).to(img1.device)

        # 국소 평균 계산
        mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
        mu2 = F.conv2d(img2, window, padding=pad, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # 국소 분산과 공분산 계산
        sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2

        # 안정화를 위한 상수
        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return ssim_map.mean() if size_average else ssim_map

    def forward(self, img1, img2, mask_pos, mask_width):
        masked_img1 = img1[:, :, :, mask_pos : mask_pos + mask_width]
        masked_img2 = img2[:, :, :, mask_pos : mask_pos + mask_width]
        ssim_value = self.ssim(
            masked_img1,
            masked_img2,
            self.window_size,
            self.size_average,
            self.val_range,
        )
        return 1 - ssim_value

    # 예시 사용법:
    # img1 = torch.rand((1, 3, 256, 256))
    # img2 = torch.rand((1, 3, 256, 256))

    # criterion = SSIMLoss()
    # loss = criterion(img1, img2)
    # print("SSIM Loss:", loss.item())


class DepthGANLoss(nn.Module):  # input: (B, 1, H, W)
    def __init__(self, use_spectral_norm=True):
        super(DepthGANLoss, self).__init__()

        def conv_block(
            in_channels, out_channels, kernel_size, stride, padding, use_norm=True
        ):
            layers = []
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            if use_spectral_norm:
                conv = utils.spectral_norm(conv)
            layers.append(conv)
            if use_norm:
                # InstanceNorm is more robust for small batch sizes
                layers.append(nn.InstanceNorm2d(out_channels, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        if use_spectral_norm:
            conv1 = utils.spectral_norm(conv1)
        layers.append(conv1)
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.extend(conv_block(64, 128, kernel_size=4, stride=2, padding=1))
        layers.extend(conv_block(128, 256, kernel_size=4, stride=2, padding=1))
        layers.extend(
            conv_block(256, 512, kernel_size=4, stride=1, padding=1, use_norm=False)
        )

        final_conv = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        if use_spectral_norm:
            final_conv = utils.spectral_norm(final_conv)
        layers.append(final_conv)
        layers.append(nn.Sigmoid())  # PatchGAN

        self.discriminator = nn.Sequential(*layers)
        self.bce_loss = nn.BCELoss()

    def forward(self, GT_depth, gen_depth, mask_pos, mask_width):

        masked_GT_depth = GT_depth[:, :, :, mask_pos : mask_pos + mask_width]
        masked_gen_depth = gen_depth[:, :, :, mask_pos : mask_pos + mask_width]

        pred_real = self.discriminator(masked_GT_depth)
        pred_fake = self.discriminator(masked_gen_depth)

        real_labels = torch.ones_like(pred_real)
        fake_labels = torch.zeros_like(pred_fake)

        loss_real = self.bce_loss(pred_real, real_labels)
        loss_fake = self.bce_loss(pred_fake, fake_labels)

        return loss_real + loss_fake

    # # 사용 예제
    # if __name__ == "__main__":
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     model = DepthGANLoss(use_spectral_norm=True).to(device)

    #     # 예제: 배치 크기 4, 1채널, 512x256 크기의 Depth Map
    #     GT_depth = torch.randn(4, 1, 512, 256).to(device)
    #     gen_depth = torch.randn(4, 1, 512, 256).to(device)

    #     loss = model(GT_depth, gen_depth)
    #     print(f"Improved Depth GAN Loss: {loss.item():.4f}")


class masked_region_MSE_Loss(nn.Module):  # input (B, C, H, W)
    def __init__(self):
        super(masked_region_MSE_Loss, self).__init__()

    def forward(self, x, y, mask_pos, mask_width):

        masked_x = torch.zeros_like(x)
        masked_y = torch.zeros_like(y)

        masked_x[:, :, :, mask_pos : mask_pos + mask_width] = x[
            :, :, :, mask_pos : mask_pos + mask_width
        ]
        masked_y[:, :, :, mask_pos : mask_pos + mask_width] = y[
            :, :, :, mask_pos : mask_pos + mask_width
        ]

        loss = F.mse_loss(masked_x, masked_y)
        return loss


class entire_region_MSE_Loss(nn.Module):  # input (B, C, H, W)
    def __init__(self):
        super(entire_region_MSE_Loss, self).__init__()

    def forward(self, x, y):
        loss = F.mse_loss(x, y)
        return loss


class Sobel_Loss(nn.Module):
    def __init__(self):
        super(Sobel_Loss, self).__init__()
        
    def get_sobel_filters(self):
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]], dtype=torch.float32)
        sobel_x = sobel_x.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
        sobel_y = sobel_y.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
        return sobel_x, sobel_y

    def compute_sobel_gradients(self, img):
        sobel_x, sobel_y = self.get_sobel_filters()
        
        B, C, H, W = img.shape
        # 각 채널별로 동일한 필터를 적용 (groups 사용)
        sobel_x = sobel_x.repeat(C, 1, 1, 1)  # (C, 1, 3, 3)
        sobel_y = sobel_y.repeat(C, 1, 1, 1)  # (C, 1, 3, 3)
        
        grad_x = F.conv2d(img, sobel_x, padding=1, groups=C)
        grad_y = F.conv2d(img, sobel_y, padding=1, groups=C)
        return grad_x, grad_y
    
    def normalize(self, masked_img1, masked_img2):
        B, C, H, W = masked_img1.shape

        masked_img1_flat = masked_img1.reshape(B, -1)
        min_val1 = masked_img1_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        max_val1 = masked_img1_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        normalized_img1 = (masked_img1 - min_val1) / (max_val1 - min_val1 + 1e-8)
        normalized_img1 = normalized_img1.reshape(B, C, H, W)

        masked_img2_flat = masked_img2.reshape(B, -1)
        min_val2 = masked_img2_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        max_val2 = masked_img2_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        normalized_img2 = (masked_img2 - min_val2) / (max_val2 - min_val2 + 1e-8)
        normalized_img2 = normalized_img2.reshape(B, C, H, W)

        return normalized_img1, normalized_img2

    def sobel_loss(self, img1, img2, mask_pos, mask_width, loss_type='L2'):
        
        masked_img1 = img1[..., mask_pos : mask_pos + mask_width]
        masked_img2 = img2[..., mask_pos : mask_pos + mask_width]
        masked_img1, masked_img2 = self.normalize(masked_img1, masked_img2)
        
        x1, y1 = self.compute_sobel_gradients(masked_img1)
        x2, y2 = self.compute_sobel_gradients(masked_img2)
        
        if loss_type == 'L1':
            loss = F.l1_loss(x1, x2) + F.l1_loss(y1, y2)
        elif loss_type == 'L2':
            loss = F.mse_loss(x1, x2) + F.mse_loss(y1, y2)
        else:
            raise ValueError("loss_type must be either 'L1' or 'L2'")
        
        return loss
    
