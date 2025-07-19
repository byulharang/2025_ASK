import torch
import torch.nn as nn
import torch.nn.functional as F


# 2번의 3x3 합성곱 + BatchNorm + ReLU 로 구성된 블록
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


# 다운샘플링 블록: maxpool 후 double conv
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.mpconv(x)


# 업샘플링 블록
class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=False):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels + skip_channels, out_channels)
        else:
            # ConvTranspose2d 후 BatchNorm을 넣어줄 수도 있으나, 여기서는 DoubleConv 내에서 BatchNorm을 적용합니다.
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # x1과 x2의 공간 크기를 맞추기 위한 padding
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # 채널 차원에서 연결
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# 최종 1x1 합성곱: 원하는 채널 수로 매핑
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# U-Net 전체 구조 (원 논문의 깊이: 인코더 4단계, 디코더 4단계)
class UNet_large(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, bilinear=True):
        """
        in_channels: 입력 이미지 채널 (예: 3)
        out_channels: 출력 이미지 채널 (예: 3)
        """
        super(UNet_large, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Contracting path
        self.inc = DoubleConv(in_channels, 64)        # (B, 64, 256, 512)
        self.down1 = Down(64, 128)                      # (B, 128, 128, 256)
        self.down2 = Down(128, 256)                     # (B, 256, 64, 128)
        self.down3 = Down(256, 512)                     # (B, 512, 32, 64)
        self.down4 = Down(512, 1024)                    # (B, 1024, 16, 32)

        # Expansive path
        self.up1 = Up(1024 + 256, 512 + 128, 512 + 128, bilinear)  # (B, 512, 32, 64)
        self.up2 = Up(512 + 128, 256 + 64, 256 + 64, bilinear)      # (B, 256, 64, 128)
        self.up3 = Up(256 + 64, 128 + 32, 128 + 32, bilinear)       # (B, 128, 128, 256)
        self.up4 = Up(128 + 32, 64, 64, bilinear)                   # (B, 64, 256, 512)

        self.outc = OutConv(64, out_channels)
              

    def forward(self, x, l5, l4, l3, l2):
        x1 = self.inc(x)      # (B, 64, 256, 512)
        x2 = self.down1(x1)   # (B, 128, 128, 256)
        x3 = self.down2(x2)   # (B, 256, 64, 128)
        x4 = self.down3(x3)   # (B, 512, 32, 64)
        x5 = self.down4(x4)   # (B, 1024, 16, 32)

        m5 = torch.cat((x5, l5), dim=1)  # (B, 1024+256, 16, 32)
        m4 = torch.cat((x4, l4), dim=1)  # (B, 512+128, 32, 64)
        m3 = torch.cat((x3, l3), dim=1)  # (B, 256+64, 64, 128)
        m2 = torch.cat((x2, l2), dim=1)  # (B, 128+?, 128, 256)  # 주석 확인: 원래 코드에서 x2는 (B,128,128,256)

        x = self.up1(m5, m4)  # (B, 512, 32, 64)
        x = self.up2(x, m3)   # (B, 256, 64, 128)
        x = self.up3(x, m2)   # (B, 128, 128, 256)
        x = self.up4(x, x1)   # (B, 64, 256, 512)

        recon_img = self.outc(x)  # (B, out_channels, 256, 512)
        return recon_img


# 모델 생성 예시
if __name__ == "__main__":
    model = UNet_large(in_channels=3, out_channels=3, bilinear=True)
    
    # 예시 입력: 배치 사이즈 1, 채널 3, 해상도 512x1024
    x = torch.randn(1, 3, 512, 1024)
    # l5, l4, l3, l2에 해당하는 feature map들도 예시로 생성합니다.
    # 각 크기는 UNet_large.forward 내 주석에 맞게 생성합니다.
    l5 = torch.randn(1, 256, 16, 32)   # x5와 concat되는 feature map
    l4 = torch.randn(1, 128, 32, 64)   # x4와 concat되는 feature map
    l3 = torch.randn(1, 64, 64, 128)    # x3와 concat되는 feature map
    l2 = torch.randn(1, 128, 128, 256)
