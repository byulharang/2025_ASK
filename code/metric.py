import torch
import lpips
import torch.nn as nn
import torch.nn.functional as F


class compute_psnr(nn.Module):
    def __init__(self, max_val=1.0):
        super(compute_psnr, self).__init__()
        self.max_val = max_val
        self.mse = nn.MSELoss()

    def normalize(self, masked_img1, masked_img2):
        B, C, H, W = masked_img1.shape

        masked_img1_flat = masked_img1.reshape(B, -1)
        min_val1 = masked_img1_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        max_val1 = masked_img1_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        masked_img1 = (masked_img1 - min_val1) / (max_val1 - min_val1 + 1e-8)

        masked_img2_flat = masked_img2.reshape(B, -1)
        min_val2 = masked_img2_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        max_val2 = masked_img2_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        masked_img2 = (masked_img2 - min_val2) / (max_val2 - min_val2 + 1e-8)

        return masked_img1, masked_img2

    def forward(self, img1, img2, mask_pos, mask_width):
        masked_img1 = img1[..., mask_pos : mask_pos + mask_width]
        masked_img2 = img2[..., mask_pos : mask_pos + mask_width]
        masked_img1, masked_img2 = self.normalize(masked_img1, masked_img2)

        mse = self.mse(masked_img1, masked_img2)
        if mse == 0:
            return float("inf")
        psnr = 10 * torch.log10((self.max_val**2) / mse)
        return psnr


class compute_lpips(nn.Module):
    def __init__(self, lpips_net="vgg", max_val=1.0):
        super(compute_lpips, self).__init__()
        self.max_val = max_val
        self.lpips_net = lpips_net
        self.lpips_model = lpips.LPIPS(net=lpips_net)
        self.lpips_model.eval()

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

    def forward(self, img1, img2, mask_pos, mask_width):
        if img1.shape[1] == 1:
            img1 = img1.repeat(1, 3, 1, 1)
        if img2.shape[1] == 1:
            img2 = img2.repeat(1, 3, 1, 1)

        masked_img1 = img1[..., mask_pos : mask_pos + mask_width]
        masked_img2 = img2[..., mask_pos : mask_pos + mask_width]
        masked_img1, masked_img2 = self.normalize(masked_img1, masked_img2)

        with torch.no_grad():
            lpips_score = self.lpips_model(masked_img1, masked_img2)
        return lpips_score


class SSIM(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, sigma)

    def gaussian(self, window_size, sigma):
        gauss = [
            torch.exp(
                torch.tensor(
                    -((x - window_size // 2) ** 2) / (2 * sigma**2), dtype=torch.float32
                )
            )
            for x in range(window_size)
        ]
        gauss = torch.stack(gauss)
        return gauss / gauss.sum()

    def create_window(self, window_size, sigma):
        _1D_window = self.gaussian(window_size, sigma).unsqueeze(1)  # (window_size, 1)
        _2D_window = _1D_window @ _1D_window.t()  # (window_size, window_size)
        _2D_window = (
            _2D_window.float().unsqueeze(0).unsqueeze(0)
        )  # (1, 1, window_size, window_size)
        return _2D_window

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

    def forward(self, img1, img2, mask_pos, mask_width):
        img1 = img1[..., mask_pos : mask_pos + mask_width]
        img2 = img2[..., mask_pos : mask_pos + mask_width]
        img1, img2 = self.normalize(img1, img2)

        # img1, img2: (N, C, H, W)
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = (
                self.create_window(self.window_size, self.sigma)
                .to(img1.device)
                .type(img1.dtype)
            )
            self.window = window
            self.channel = channel

        window = window.expand(channel, 1, self.window_size, self.window_size)

        # 계산: local mean
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # 계산: variance, covariance
        sigma1_sq = (
            F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel)
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel)
            - mu2_sq
        )
        sigma12 = (
            F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel)
            - mu1_mu2
        )

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        if self.size_average:
            return ssim_map.mean()
        else:
            # 각 채널별 SSIM map 리턴 (N, C, H, W)
            return ssim_map

    # # 사용 예시:
    # if __name__ == '__main__':
    #     # 임의의 이미지 생성 (batch=4, channel=3, height=256, width=256)
    #     img1 = torch.rand(4, 3, 256, 256)
    #     img2 = torch.rand(4, 3, 256, 256)

    #     ssim_metric = SSIM(window_size=11, sigma=1.5, size_average=True)
    #     ssim_value = ssim_metric(img1, img2)
    #     print("SSIM:", ssim_value.item())


class RelativeError(nn.Module):
    def __init__(self, eps=1e-8):
        super(RelativeError, self).__init__()
        self.eps = eps

    def normalize(self, masked_img1, masked_img2):
        B, C, H, W = masked_img1.shape

        masked_img1_flat = masked_img1.reshape(B, -1)
        min_val1 = masked_img1_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        max_val1 = masked_img1_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        masked_img1 = (masked_img1 - min_val1) / (max_val1 - min_val1 + 1e-8)

        masked_img2_flat = masked_img2.reshape(B, -1)
        min_val2 = masked_img2_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        max_val2 = masked_img2_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        masked_img2 = (masked_img2 - min_val2) / (max_val2 - min_val2 + 1e-8)

        return masked_img1, masked_img2

    def forward(self, GT, gen, mask_pos, mask_width):
        partial_GT = GT[..., mask_pos : mask_pos + mask_width]
        partial_GEN = gen[..., mask_pos : mask_pos + mask_width]

        rel = torch.mean(
            torch.abs(partial_GT - partial_GEN) / (torch.abs(partial_GT) + self.eps)
        )
        if rel > 10:
            return torch.tensor(1.0).to(device=rel.device)
        return rel
