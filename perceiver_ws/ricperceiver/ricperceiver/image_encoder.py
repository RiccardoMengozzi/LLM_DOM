import torch
import torch.nn as nn
from torch.nn import functional as F
from ricperceiver.resnet import resnet34, resnet50, resnet101


class Backbone(nn.Module):

    def __init__(self, resnet="resnet101", feature_dim=256, input_dim=3):
        super(Backbone, self).__init__()
        if resnet == "resnet34":
            self.backbone = resnet34(
                pretrained=False,
                replace_stride_with_dilation=[False, False, False],
                num_classes=feature_dim,
                input_dim=input_dim,
            )
            dims = [64, 128, 256, 512]
        elif resnet == "resnet50":
            self.backbone = resnet50(
                pretrained=False,
                replace_stride_with_dilation=[False, False, True],
                num_classes=feature_dim,
                input_dim=input_dim,
            )
            dims = [256, 512, 1024, 2048]
        elif resnet == "resnet101":
            self.backbone = resnet101(
                pretrained=False,
                replace_stride_with_dilation=[False, False, True],
                num_classes=feature_dim,
                input_dim=input_dim,
            )
            dims = [256, 512, 1024, 2048]
        else:
            raise ValueError("Invalid resnet type")

        self.conv1 = nn.Conv2d(dims[0], feature_dim // 4, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(dims[1], feature_dim // 4, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(dims[2], feature_dim // 4, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(dims[3], feature_dim // 4, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1, x2, x3, x4, xfc = self.backbone(x)

        x1 = self.conv1(x1).relu()
        x2 = self.conv2(x2).relu()
        x3 = self.conv3(x3).relu()
        x4 = self.conv4(x4).relu()

        x1_down = F.interpolate(x1, scale_factor=1/4, mode="bilinear", align_corners=False)
        x2_down = F.interpolate(x2, scale_factor=1/2, mode="bilinear", align_corners=False)


        x_cat = torch.cat([x1_down, x2_down, x3, x4], dim=1)
        return x_cat
