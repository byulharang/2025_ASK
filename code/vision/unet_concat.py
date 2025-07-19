import torch
import torch.nn as nn
import torch.nn.functional as F


# 2번의 3x3 합성곱(ReLU)로 구성된 블록
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
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
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
            # in_channels: 낮은 해상도 feature map의 채널 수
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2
            )
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
class UNet_concat(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, bilinear=False):
        """
        in_channels: 입력 이미지 채널 (여기서는 3)
        out_channels: 출력 이미지 채널 (여기서는 복원을 위해 3)
        """
        super(UNet_concat, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Contracting path
        self.inc = DoubleConv(in_channels, 64)  # 첫 번째 레벨
        self.down1 = Down(64, 128)  # 두 번째 레벨
        self.down2 = Down(128, 256)  # 세 번째 레벨
        self.down3 = Down(256, 512)  # 네 번째 레벨
        self.down4 = Down(512, 1024)  # Bottleneck

        # Expansive path (올바른 채널 수를 인자로 전달)
        self.up1 = Up(1024 + 256, 512 + 128, 512 + 128, bilinear)
        self.up2 = Up(512 + 128, 256 + 64, 256+64, bilinear)
        self.up3 = Up(256+64, 128+32, 128+32, bilinear)
        self.up4 = Up(128+32, 64, 64, bilinear)

        self.outc = OutConv(64, out_channels)  # 최종 1x1 합성곱

    def forward(self, x, l5, l4, l3, l2):
        x1 = self.inc(x)  # O/P: (B, 64, 256, 512)
        x2 = self.down1(x1)  # (B, 128, 128, 256)
        x3 = self.down2(x2)  # (B, 256, 64, 128)
        x4 = self.down3(x3)  # (B, 512, 32, 64)
        x5 = self.down4(x4)  # (B, 1024, 16, 32)

        m5 = torch.cat((x5, l5), dim=1)  # (B, 1024+256, 16, 32)
        m4 = torch.cat((x4, l4), dim=1)  # (B, 512+128, 32, 64)
        m3 = torch.cat((x3, l3), dim=1)  # (B, 256+64, 64, 128)
        m2 = torch.cat((x2, l2), dim=1)  # (B, 512+128, 32, 64)

        x = self.up1(m5, m4)  # (B, 512, 32, 64)
        x = self.up2(x, m3)  # (B, 256, 64, 128)
        x = self.up3(x, m2)  # (B, 128, 128, 256)
        x = self.up4(x, x1)  # (B, 64, 256, 512)
        
        # print("x5: ", x5.max(), x5.min(), "l5: ", l5.max(), l5.min())
        # print("x4: ", x4.max(), x4.min(), "l4: ", l4.max(), l4.min())
        # print("x3: ", x3.max(), x3.min(), "l3: ", l3.max(), l3.min())
        # print("x2: ", x2.max(), x2.min(), "l2: ", l2.max(), l2.min())
        
        recon_img = self.outc(x)  # (B, 1, 256, 512)
        return recon_img


# 모델 생성 예시
if __name__ == "__main__":
    # bilinear=True 또는 False에 따라 업샘플링 방식이 달라집니다.
    model = UNet_concat(in_channels=3, out_channels=3, bilinear=True)
    x = torch.randn(1, 3, 512, 1024)  # 배치 사이즈 1의 예시 입력
    y = model(x)
    print(y.shape)  # 결과: torch.Size([1, 3, 512, 1024])
