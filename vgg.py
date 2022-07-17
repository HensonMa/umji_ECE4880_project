import torch
import torch.nn as nn

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(nn.Module):

    def __init__(self, features, num_class=100, dataset="cifar10"):
        super().__init__()
        self.features = features

        if dataset == "cifar10":
            self.classifier = nn.Sequential(
                nn.Linear(512, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, 10),
            )
        elif dataset == "imagenet":
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, num_class)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        output = self.features(x)
        output = torch.flatten(output, 1)
        output = self.classifier(output)

        return output


def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=(3, 3), padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)


def vgg11_bn():
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13_bn():
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16_bn():
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19_bn():
    return VGG(make_layers(cfg['E'], batch_norm=True))


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )


class vgg19(nn.Module):

    def __init__(self, num_class=10, dataset="cifar10"):
        super(vgg19, self).__init__()
        self.init_weights = True
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

    def forward(self, x):
        out = self.pool1(x)
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
        return out
