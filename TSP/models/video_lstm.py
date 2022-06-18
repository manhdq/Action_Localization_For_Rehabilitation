from typing import Tuple, Optional, Callable, List, Sequence, Type, Any, Union

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["r10_lstm_2l", "r10_bilstm_2l"]


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicStem(nn.Sequential):
    """The default conv-batchnorm-relu stem"""

    def __init__(self, num_channels=3) -> None:
        super().__init__(
            nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# class ResNet(nn.Module):
#     def __init__(
#         self,
#         block,
#         layers: List[int],
#         num_classes: int = 1000,
#         zero_init_residual: bool = False,
#         groups: int = 1,
#         width_per_group: int = 64,
#         replace_stride_with_dilation: Optional[List[bool]] = None,
#         norm_layer: Optional[Callable[..., nn.Module]] = None,
#     ) -> None:
#         super().__init__()
#         _log_api_usage_once(self)
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self._norm_layer = norm_layer

#         self.inplanes = 64
#         self.dilation = 1
#         if replace_stride_with_dilation is None:
#             # each element in the tuple indicates if we should replace
#             # the 2x2 stride with a dilated convolution instead
#             replace_stride_with_dilation = [False, False, False]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError(
#                 "replace_stride_with_dilation should be None "
#                 f"or a 3-element tuple, got {replace_stride_with_dilation}"
#             )
#         self.groups = groups
#         self.base_width = width_per_group

#         self.stem = stem()

#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#         # Zero-initialize the last BN in each residual branch,
#         # so that the residual branch starts with zeros, and each residual block behaves like an identity.
#         # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Bottleneck) and m.bn3.weight is not None:
#                     nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
#                 elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
#                     nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

#     def _make_layer(
#         self,
#         block: Type[Union[BasicBlock, Bottleneck]],
#         planes: int,
#         blocks: int,
#         stride: int = 1,
#         dilate: bool = False,
#     ) -> nn.Sequential:
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 norm_layer(planes * block.expansion),
#             )

#         layers = []
#         layers.append(
#             block(
#                 self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
#             )
#         )
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(
#                 block(
#                     self.inplanes,
#                     planes,
#                     groups=self.groups,
#                     base_width=self.base_width,
#                     dilation=self.dilation,
#                     norm_layer=norm_layer,
#                 )
#             )

#         return nn.Sequential(*layers)

#     def _forward_impl(self, x: Tensor) -> Tensor:
#         # See note [TorchScript super()]
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)

#         return x

#     def forward(self, x: Tensor) -> Tensor:
#         return self._forward_impl(x)



class VideoLSTM(nn.Module):
    def __init__(
        self,
        block,
        layers: List[int],
        stem: Callable[..., nn.Module],
        num_classes: int = 400,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        # LSTM setup
        hidden_dim: int = 128,
        n_layers: int = 2,
        bidirectional: bool = False,
        num_channels: int = 3,
        **kwargs
    ) -> None:
        """Generic resnet video generator.

        Args:
            block (Type[Union[BasicBlock, Bottleneck]]): resnet building block
            conv_makers (List[Type[Union[Conv3DSimple, Conv3DNoTemporal, Conv2Plus1D]]]): generator
                function for each layer
            layers (List[int]): number of blocks per layer
            stem (Callable[..., nn.Module]): module specifying the ResNet stem.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        self.D = 2 if bidirectional else 1

        self.stem = stem(num_channels)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.lstm = nn.LSTM(128 * block.expansion,
                            hidden_dim,
                            n_layers,
                            dropout=.2,
                            batch_first=True,
                            bidirectional=bidirectional)

        fc_layers = []
        fc_layers.append(nn.Dropout(.2))
        fc_layers.append(nn.Linear(self.hidden_dim * self.D * 2, 256))
        fc_layers.append(nn.Dropout(.2))
        fc_layers.append(nn.Linear(256, 128))
        self.fc_layers = nn.Sequential(*fc_layers)

        self.fc = nn.Linear(128, num_classes)

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def forward(self, x: Tensor, hidden: Tensor) -> Tensor:
        # x: [batch, channels, sequences, height, width]
        batch_size = x.shape[0]
        num_channels = x.shape[1]
        num_seqs = x.shape[2]
        height = x.shape[3]
        width = x.shape[4]

        x = x.permute(0, 2, 1, 3, 4).contiguous()

        x = x.view(batch_size*num_seqs, num_channels, height, width)
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        # Flatten the layer to lstm
        x = x.flatten(1)
        x = x.view(batch_size, num_seqs, -1)
        
        lstm_out, hidden = self.lstm(x, hidden)
        encoding = torch.cat((lstm_out[:, 0], lstm_out[:, -1]), dim=1)
        
        features = self.fc_layers(encoding)

        out = self.fc(features)

        return out, hidden

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers * self.D, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers * self.D, batch_size, self.hidden_dim).zero_().to(device))

        return hidden

    def _make_layer(
        self,
        block,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

def _video_lstm(**kwargs: Any) -> VideoLSTM:
    model = VideoLSTM(**kwargs)
    return model


def r10_lstm_2l(hidden_dim=128, **kwargs: Any) -> VideoLSTM:
    """Constructor for the LSTM with 2 layers

    Args:
        hidden_dim (int): hidden dim in fully connected

    Returns:
        nn.Module: R(2+1)D-18 network
    """
    return _video_lstm(
        block=BasicBlock,
        layers=[2, 2],
        zero_init_residual=True,
        hidden_dim=hidden_dim,
        n_layers=2,
        bidirectional=False,
        stem=BasicStem,
        **kwargs,
    )

def r10_bilstm_2l(hidden_dim=128, **kwargs: Any) -> VideoLSTM:
    """Constructor for the LSTM with 2 layers

    Args:
        hidden_dim (int): hidden dim in fully connected

    Returns:
        nn.Module: R(2+1)D-18 network
    """
    return _video_lstm(
        block=BasicBlock,
        layers=[2, 2],
        zero_init_residual=True,
        hidden_dim=hidden_dim,
        n_layers=2,
        bidirectional=True,
        stem=BasicStem,
        **kwargs,
    )


if __name__ == '__main__':
    model = r10_bilstm_2l(num_channels=5)

    h = model.init_hidden(2, 'cpu')
    x = torch.randn(2, 5, 4, 128, 230)  # b, c, s, h, w
    print(model(x, h))