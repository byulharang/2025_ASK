import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18_pyramidal(nn.Module):
    def __init__(
        self,
        dropout_prob=0.0,
        in_channel=8,
        intermediate_channel=64,
        out_dim=512,
        out_channel=256,
        out_size=(32, 64),
        pretrained=None,
        spatial_feature=True,
    ):
        super(ResNet18_pyramidal, self).__init__()

        self.model = models.resnet18()
        self.model.conv1 = nn.Conv2d(
            in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.p = dropout_prob
        self.out_features = out_dim
        self.out_size = out_size
        self.intermediate_channel = intermediate_channel
        self.out_channel = out_channel
        self.spatial_feature = spatial_feature

        self.model.layer1 = self._add_dropout(self.model.layer1)
        self.model.layer2 = self._add_dropout(self.model.layer2)
        self.model.layer3 = self._add_dropout(self.model.layer3)
        self.model.layer4 = self._add_dropout(self.model.layer4)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=self.p), nn.Linear(in_features, self.out_features)
        )

        # Channel adjustment layers for each stage
        self.channel_adjust_layers = nn.ModuleList(
            [
                nn.Conv2d(
                    64,
                    self.intermediate_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.Conv2d(
                    128,
                    self.intermediate_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.Conv2d(
                    256,
                    self.intermediate_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.Conv2d(
                    512,
                    self.intermediate_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
            ]
        )

        # Adaptive Pooling layers to ensure consistent output size
        self.adaptive_pool_layers = nn.ModuleList(
            [
                nn.AdaptiveAvgPool2d(self.out_size),
                nn.AdaptiveAvgPool2d(self.out_size),
                nn.AdaptiveAvgPool2d(self.out_size),
                nn.AdaptiveAvgPool2d(self.out_size),
            ]
        )

    def _add_dropout(self, layer):
        for name, module in layer.named_children():
            layer.add_module(name, nn.Sequential(module, nn.Dropout(p=self.p)))
        return layer

    def forward(self, x):
        outputs = []

        # Initial layers
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        # Layer 1
        x = self.model.layer1(x)
        adjusted = self.channel_adjust_layers[0](x)
        pooled = self.adaptive_pool_layers[0](adjusted)
        outputs.append(pooled)

        # Layer 2
        x = self.model.layer2(x)
        adjusted = self.channel_adjust_layers[1](x)
        pooled = self.adaptive_pool_layers[1](adjusted)
        outputs.append(pooled)

        # Layer 3
        x = self.model.layer3(x)
        adjusted = self.channel_adjust_layers[2](x)
        pooled = self.adaptive_pool_layers[2](adjusted)
        outputs.append(pooled)

        # Layer 4
        x = self.model.layer4(x)
        adjusted = self.channel_adjust_layers[3](x)
        pooled = self.adaptive_pool_layers[3](adjusted)
        outputs.append(pooled)

        concatenated_features = torch.cat(outputs, dim=1)
        return concatenated_features
