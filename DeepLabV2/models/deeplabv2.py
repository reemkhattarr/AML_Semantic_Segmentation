# deeplabv2.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101, ResNet101_Weights
from typing import Iterator, Dict, Any, Optional

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) as in DeepLabV2 [[7]].
    """
    def __init__(self, in_channels: int, out_channels: int, rates: list[int]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=True)
            for rate in rates
        ])
        self._init_weights()

    def _init_weights(self):
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sum(conv(x) for conv in self.convs)


class DeepLabV2(nn.Module):
    """
    DeepLabV2 with ResNet-101 backbone and ASPP, following [[7]].
    - Supports batch norm freezing and optimizer parameter groups for 1x/10x LR.
    - Adapted for research reproducibility and ablation.
    """
    def __init__(
        self,
        num_classes: int = 7,
        pretrained_backbone: bool = True,
        freeze_bn: bool = True,
    ):
        super().__init__()
        
        # Load backbone
        resnet = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1 if pretrained_backbone else None, replace_stride_with_dilation=[False, True, True])
        # Remove avgpool/fc, keep up to layer4
        self.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        '''
        self.layer3 = self._convert_to_dilated(resnet.layer3, dilation=2)
        self.layer4 = self._convert_to_dilated(resnet.layer4, dilation=4)
        '''
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.aspp = ASPP(2048, num_classes, rates=[6,12,18,24])

        if freeze_bn:
            self.freeze_batchnorm()
    
    def _convert_to_dilated(self, layer: nn.Sequential, dilation: int) -> nn.Sequential:
        for n, m in layer.named_modules():
            if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
                m.dilation = (dilation, dilation)
                m.padding = (dilation, dilation)
                m.stride = (1, 1)
            # Also patch the downsample path
            if isinstance(m, Bottleneck):
                if m.downsample is not None:
                    for dn in m.downsample:
                        if isinstance(dn, nn.Conv2d):
                            dn.stride = (1, 1)
        return layer

    def freeze_batchnorm(self):
        """
        Set all BatchNorm layers to eval mode and freeze their parameters.
        This matches DeepLabV2 best practice for segmentation (see [[7]], appendix).
        """
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.aspp(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        return x

    def get_1x_lr_params(self) -> Iterator[nn.Parameter]:
        """
        Parameters of the backbone for 1x learning rate.
        """
        for name, module in [
            ('layer0', self.layer0), ('layer1', self.layer1),
            ('layer2', self.layer2), ('layer3', self.layer3), ('layer4', self.layer4)
        ]:
            for param in module.parameters():
                if param.requires_grad:
                    yield param

    def get_10x_lr_params(self) -> Iterator[nn.Parameter]:
        """
        Parameters of the ASPP head for 10x learning rate.
        """
        for param in self.aspp.parameters():
            if param.requires_grad:
                yield param

    def optim_parameters(self, lr: float) -> list[Dict[str, Any]]:
        """
        Suggest parameter groups for optimizer: 1x for backbone, 10x for ASPP.
        """
        return [
            {"params": self.get_1x_lr_params(), "lr": lr},
            {"params": self.get_10x_lr_params(), "lr": lr * 10}
        ]

def get_deeplabv2_model(
    num_classes: int,
    pretrained_backbone: bool = True,
    freeze_bn: bool = True,
) -> DeepLabV2:
    """
    Helper to instantiate DeepLabV2 with pretrained backbone and batchnorm freezing.
    """
    model = DeepLabV2(num_classes, pretrained_backbone, freeze_bn)
    return model

# Example usage (in train.py or a notebook):
# model = get_deeplabv2_model(num_classes=7)
# optimizer = torch.optim.SGD(model.optim_parameters(lr=0.01), momentum=0.9, weight_decay=1e-4)
