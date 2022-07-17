import torch
import torch.quantization as tq
from torch.quantization import QuantStub, DeQuantStub
import torch.nn as nn
import os
from vgg import *
import time


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )


class vgg19_quantized(nn.Module):

    def __init__(self, num_class=10, dataset="cifar10"):
        super(vgg19_quantized, self).__init__()
        self.in_channels = 3
        self.num_class = num_class

        self.conv1 = nn.Sequential(
            ConvBNReLU(self.in_channels, 64, kernel_size=(3, 3), stride=1, padding=1),
            ConvBNReLU(64, 64, kernel_size=(3, 3), stride=1, padding=1),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            ConvBNReLU(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            ConvBNReLU(128, 128, kernel_size=(3, 3), stride=1, padding=1),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            ConvBNReLU(128, 256, kernel_size=(3, 3), stride=1, padding=1),
            ConvBNReLU(256, 256, kernel_size=(3, 3), stride=1, padding=1),
            ConvBNReLU(256, 256, kernel_size=(3, 3), stride=1, padding=1),
            ConvBNReLU(256, 256, kernel_size=(3, 3), stride=1, padding=1),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(
            ConvBNReLU(256, 512, kernel_size=(3, 3), stride=1, padding=1),
            ConvBNReLU(512, 512, kernel_size=(3, 3), stride=1, padding=1),
            ConvBNReLU(512, 512, kernel_size=(3, 3), stride=1, padding=1),
            ConvBNReLU(512, 512, kernel_size=(3, 3), stride=1, padding=1),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Sequential(
            ConvBNReLU(512, 512, kernel_size=(3, 3), stride=1, padding=1),
            ConvBNReLU(512, 512, kernel_size=(3, 3), stride=1, padding=1),
            ConvBNReLU(512, 512, kernel_size=(3, 3), stride=1, padding=1),
            ConvBNReLU(512, 512, kernel_size=(3, 3), stride=1, padding=1),
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        if dataset == "cifar10" or "cifar100":
            self.classifier = nn.Sequential(
                nn.Linear(512, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, self.num_class),
            )
        elif dataset == "imagenet":
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, self.num_class)
            )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)

    def forward(self, x):
        out = self.quant(x)
        out = self.pool1(out)
        out = self.conv1(out)

        out = self.pool2(out)
        out = self.conv2(out)

        out = self.pool3(out)
        out = self.conv3(out)

        out = self.pool4(out)
        out = self.conv4(out)

        out = self.pool5(out)
        out = self.conv5(out)

        out = torch.flatten(out, 1)
        out = self.classifier(out)
        out = self.dequant(out)
        return out


class vgg19_quantized_partial(nn.Module):

    def __init__(self, num_class=10, dataset="cifar10", begin=1):
        super(vgg19_quantized_partial, self).__init__()
        self.in_channels = 3
        self.num_class = num_class
        self.begin = begin

        self.conv1 = nn.Sequential(
            ConvBNReLU(self.in_channels, 64, kernel_size=(3, 3), stride=1, padding=1),
            ConvBNReLU(64, 64, kernel_size=(3, 3), stride=1, padding=1),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            ConvBNReLU(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            ConvBNReLU(128, 128, kernel_size=(3, 3), stride=1, padding=1),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            ConvBNReLU(128, 256, kernel_size=(3, 3), stride=1, padding=1),
            ConvBNReLU(256, 256, kernel_size=(3, 3), stride=1, padding=1),
            ConvBNReLU(256, 256, kernel_size=(3, 3), stride=1, padding=1),
            ConvBNReLU(256, 256, kernel_size=(3, 3), stride=1, padding=1),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(
            ConvBNReLU(256, 512, kernel_size=(3, 3), stride=1, padding=1),
            ConvBNReLU(512, 512, kernel_size=(3, 3), stride=1, padding=1),
            ConvBNReLU(512, 512, kernel_size=(3, 3), stride=1, padding=1),
            ConvBNReLU(512, 512, kernel_size=(3, 3), stride=1, padding=1),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Sequential(
            ConvBNReLU(512, 512, kernel_size=(3, 3), stride=1, padding=1),
            ConvBNReLU(512, 512, kernel_size=(3, 3), stride=1, padding=1),
            ConvBNReLU(512, 512, kernel_size=(3, 3), stride=1, padding=1),
            ConvBNReLU(512, 512, kernel_size=(3, 3), stride=1, padding=1),
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        if dataset == "cifar10" or "cifar100":
            self.classifier = nn.Sequential(
                nn.Linear(512, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, self.num_class),
            )
        elif dataset == "imagenet":
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, self.num_class)
            )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)

    def forward(self, x):
        out = self.pool1(x)
        out = self.conv1(out)

        if self.begin == 1:
            out = self.quant(out)
        out = self.pool2(out)
        out = self.conv2(out)

        if self.begin == 2:
            out = self.quant(out)
        out = self.pool3(out)
        out = self.conv3(out)

        if self.begin == 3:
            out = self.quant(out)
        out = self.pool4(out)
        out = self.conv4(out)

        if self.begin == 4:
            out = self.quant(out)
        out = self.pool5(out)
        out = self.conv5(out)

        out = torch.flatten(out, 1)
        out = self.classifier(out)
        out = self.dequant(out)
        return out