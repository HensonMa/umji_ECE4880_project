import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import Int8Bias as BiasQuant
import torch.nn.functional as F

cfg = {
    'A': [[64, 'M'], [128, 'M'], [256, 256, 'M'], [512, 512, 'M'], [512, 512, 'M']],
    'B': [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 'M'], [512, 512, 'M'], [512, 512, 'M']],
    'D': [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 256, 'M'], [512, 512, 512, 'M'], [512, 512, 512, 'M']],
    'E': [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 256, 256, 'M'], [512, 512, 512, 512, 'M'], [512, 512, 512, 512, 'M']]
}

class VGG_quantized(nn.Module):

    def __init__(self, cfg_d, num_class=100, dataset="cifar10", batch_norm=False, bit_width=8, depth=[1, 2, 3, 4, 5]):
        super().__init__()
        self.cfg = cfg_d
        self.bit_width = bit_width
        self.num_class = num_class
        self.bn = batch_norm
        self.depth = depth

        self.quant_inp = qnn.QuantIdentity(bit_width=self.bit_width, return_quant_tensor=True)

        self.conv1, in_ch = self.make_layers(self.cfg[0], 3, 1 in self.depth)
        self.conv2, in_ch = self.make_layers(self.cfg[1], in_ch, 2 in self.depth)
        self.conv3, in_ch = self.make_layers(self.cfg[2], in_ch, 3 in self.depth)
        self.conv4, in_ch = self.make_layers(self.cfg[3], in_ch, 4 in self.depth)
        self.conv5, _ = self.make_layers(self.cfg[4], in_ch, 5 in self.depth)

        if dataset == "cifar10":
            self.classifier = nn.Sequential(
                qnn.QuantLinear(512, 4096, bias=True, weight_bit_width=self.bit_width, bias_quant=BiasQuant, return_quant_tensor=True),
                qnn.QuantReLU(bit_width=8, return_quant_tensor=True),
                nn.Dropout(0.5),
                qnn.QuantLinear(4096, 4096, bias=True, weight_bit_width=self.bit_width, bias_quant=BiasQuant, return_quant_tensor=True),
                qnn.QuantReLU(bit_width=8, return_quant_tensor=True),
                nn.Dropout(0.5),
                qnn.QuantLinear(4096, self.num_class, bias=True, weight_bit_width=self.bit_width, bias_quant=BiasQuant),
            )
        elif dataset == "imagenet":
            self.classifier = nn.Sequential(
                qnn.QuantLinear(512 * 7 * 7, 4096, bias=True, weight_bit_width=self.bit_width, bias_quant=BiasQuant, return_quant_tensor=True),
                qnn.QuantReLU(bit_width=8, return_quant_tensor=True),
                nn.Dropout(0.5),
                qnn.QuantLinear(4096, 4096, bias=True, weight_bit_width=self.bit_width, bias_quant=BiasQuant, return_quant_tensor=True),
                qnn.QuantReLU(bit_width=8, return_quant_tensor=True),
                nn.Dropout(0.5),
                qnn.QuantLinear(4096, self.num_class, bias=True, weight_bit_width=self.bit_width, bias_quant=BiasQuant),
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

    def make_layers(self, layer_cfg, input_channel, quantized=False):
        layers = []

        for l in layer_cfg:
            if l == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                continue

            if quantized:
                layers += [qnn.QuantConv2d(input_channel, l, 3, weight_bit_width=self.bit_width, padding=1, bias_quant=BiasQuant, return_quant_tensor=True)]
            else:
                layers += [nn.Conv2d(input_channel, l, kernel_size=(3, 3), padding=1)]

            if self.bn:
                layers += [nn.BatchNorm2d(l)]

            if quantized:
                layers += [qnn.QuantReLU(bit_width=self.bit_width, return_quant_tensor=True)]
            else:
                layers += [nn.ReLU(inplace=True)]

            input_channel = l

        return nn.Sequential(*layers), input_channel

    def forward(self, x):
        first_idx = min(self.depth)

        if first_idx == 1:
            x = self.quant_inp(x)
        x = self.conv1(x)
        if first_idx == 2:
            x = self.quant_inp(x)
        x = self.conv2(x)
        if first_idx == 3:
            x = self.quant_inp(x)
        x = self.conv3(x)
        if first_idx == 4:
            x = self.quant_inp(x)
        x = self.conv4(x)
        if first_idx == 5:
            x = self.quant_inp(x)
        x = self.conv5(x)

        x = torch.flatten(x, 1)
        output = self.classifier(x)

        return output


def vgg19_quantized(num_class=10, dataset="cifar10", batch_norm=False, bit_width=8, depth=[1, 2, 3, 4, 5]):
    return VGG_quantized(cfg['E'], num_class=num_class, dataset=dataset, batch_norm=batch_norm, bit_width=bit_width, depth=[1, 2, 3, 4, 5])

# model = vgg19_quantized(batch_norm=True, depth=[3, 4, 5])
# print(model(torch.randn(2, 3, 32, 32)).shape)
