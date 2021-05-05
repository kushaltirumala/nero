"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResBlockv2(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1,
                 affine=True,bn_flag=True, nl_flag=True, pr_flag=False,
                 bias_flag=False,res_flag=True,scale=1.0,momentum=0.1,
                 init_flag=True,beta=1.0):
        super().__init__()
        bias = bias_flag
        self.scale = scale
        self.res_flag = res_flag
        self.pr_flag = pr_flag
        self.bn_flag = bn_flag
        self.nl_flag = nl_flag
        self.init_flag = init_flag
        self.beta = 1.0 #beta

        if self.nl_flag:
            self.nl = nn.ReLU(inplace=False)

        if self.bn_flag:
            self.bn1 = nn.BatchNorm2d(in_channels, affine=affine,momentum=momentum)
            self.bn2 = nn.BatchNorm2d(out_channels, affine=affine,momentum=momentum)

        self.conv1 = nn.Conv2d(in_channels, out_channels,kernel_size=3, padding=1, stride=stride,bias=bias)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3, padding=1, stride=1,bias=bias)

        if self.init_flag:
            print('custom init')
            self.conv1.weight.data -= neuron_mean(self.conv1.weight.data)
            self.conv1.weight.data /= neuron_norm(self.conv1.weight.data)
            self.conv1.weight.data *= math.sqrt(2/(1-1/math.pi))
            self.conv2.weight.data -= neuron_mean(self.conv2.weight.data)
            self.conv2.weight.data /= neuron_norm(self.conv2.weight.data)
            self.conv2.weight.data *= math.sqrt(2/(1-1/math.pi))

        self.identity = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.identity = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
            if self.init_flag:
                self.identity.weight.data -= neuron_mean(self.identity.weight.data)
                self.identity.weight.data /= neuron_norm(self.identity.weight.data)
                self.identity.weight.data *= math.sqrt(2/(1-1/math.pi))

    def forward(self, x):
        #x = x/self.beta
        if self.res_flag:
            identity = self.identity(x)
            if self.pr_flag:
                print('idendity: chn #: {}, mean: {:0.3f}, std: {:0.3f}\n'.format(identity.shape, float(torch.mean(torch.mean(identity,dim=(0,2,3)))), float(torch.mean(torch.std(identity,dim=(0,2,3)))) ) )
        if self.bn_flag:
            x = self.bn1(x)
            if self.pr_flag:
                print('bn1 out: chn #: {}, mean: {:0.3f}, std: {:0.3f}\n'.format(x.shape, float(torch.mean(torch.mean(x,dim=(0,2,3)))), float(torch.mean(torch.std(x,dim=(0,2,3)))) ) )
            x = self.nl(x)
            x = self.conv1(x)
            if self.pr_flag:
                print('conv1 out: chn #: {}, mean: {:0.3f}, std: {:0.3f}\n'.format(x.shape, float(torch.mean(torch.mean(x,dim=(0,2,3)))), float(torch.mean(torch.std(x,dim=(0,2,3)))) ) )
            x = self.bn2(x)
            if self.pr_flag:
                print('bn2 out: chn #: {}, mean: {:0.3f}, std: {:0.3f}\n'.format(x.shape, float(torch.mean(torch.mean(x,dim=(0,2,3)))), float(torch.mean(torch.std(x,dim=(0,2,3)))) ) )
            x = self.nl(x)
            x = self.conv2(x)
            if self.pr_flag:
                print('conv2 out: chn #: {}, mean: {:0.3f}, std: {:0.3f}\n'.format(x.shape, float(torch.mean(torch.mean(x,dim=(0,2,3)))), float(torch.mean(torch.std(x,dim=(0,2,3)))) ) )
        else:
            x = self.nl(x)
            x = self.conv1(x)
            if self.pr_flag:
                print('conv1 out: chn #: {}, mean: {:0.3f}, std: {:0.3f}\n'.format(x.shape, float(torch.mean(torch.mean(x,dim=(0,2,3)))), float(torch.mean(torch.std(x,dim=(0,2,3)))) ) )
            x = self.nl(x)
            x = self.conv2(x)
            if self.pr_flag:
                print('conv2 out: chn #: {}, mean: {:0.3f}, std: {:0.3f}\n'.format(x.shape, float(torch.mean(torch.mean(x,dim=(0,2,3)))), float(torch.mean(torch.std(x,dim=(0,2,3)))) ) )

        if self.res_flag:
            x = identity + x

        if self.pr_flag:
            print('block out: chn #: {}, mean: {:0.3f}, std: {:0.3f}\n'.format(x.shape, float(torch.mean(torch.mean(x,dim=(0,2,3)))), float(torch.mean(torch.std(x,dim=(0,2,3)))) ) )
        return x


class ResNet(nn.Module):
    def __init__(self, block=ResBlockv2, width=64, depth=10,
                 bn_flag=True, bias_flag=True, pr_flag=False,
                 res_flag=True, init_flag=False,
                 num_classes=10):
        super().__init__()
        assert (depth - 2) % 8 == 0 and (depth - 2) // 8 > 0
        num_block = (depth - 2) // 8
        self.pr_flag = pr_flag
        self.bias_flag = bias_flag
        self.init_flag = init_flag
        self.input_channels = width
        self.exp_var = 1.0
        self.conv0 = nn.Conv2d(3, int(width), 3, padding=1)

        self.stage1 = self._make_layers(block, num_block, int(width),  1,
                                        bn_flag, res_flag, pr_flag, init_flag,bias_flag)
        self.stage2 = self._make_layers(block, num_block, int(width*2), 2,
                                        bn_flag, res_flag, pr_flag, init_flag,bias_flag)
        self.stage3 = self._make_layers(block, num_block, int(width*4), 2,
                                        bn_flag, res_flag, pr_flag, init_flag,bias_flag)
        self.stage4 = self._make_layers(block, num_block, int(width*8), 2,
                                        bn_flag, res_flag, pr_flag, init_flag,bias_flag)

        self.linear = nn.Linear(self.input_channels, num_classes)
        if self.init_flag:
            print('body custom init!')
            self.conv0.weight.data -= neuron_mean(self.conv0.weight.data)
            self.conv0.weight.data /= neuron_norm(self.conv0.weight.data)
            self.conv0.weight.data *= math.sqrt(2/(1-1/math.pi))
            self.linear.weight.data -= neuron_mean(self.linear.weight.data)
            self.linear.weight.data /= neuron_norm(self.linear.weight.data)

    def _make_layers(self, block, block_num, out_channels, stride,
                     bn_flag,res_flag,pr_flag,init_flag,bias_flag):
        layers = []
        #self.exp_var = 1.0
        beta = self.exp_var ** 0.5
        layers.append(block(in_channels=self.input_channels,
                            out_channels=out_channels,
                            stride=stride,
                            bn_flag=bn_flag,
                            res_flag=res_flag,
                            pr_flag=pr_flag,
                            beta=beta,
                            init_flag=init_flag,
                            bias_flag=bias_flag))
        self.input_channels = out_channels * block.expansion
        self.exp_var += 1

        while block_num - 1:
            beta = self.exp_var ** 0.5
            # print(beta)
            layers.append(block(in_channels=self.input_channels,
                            out_channels=out_channels,
                            stride=1,
                            bn_flag=bn_flag,
                            res_flag=res_flag,
                            pr_flag=pr_flag,
                            beta=beta,
                            init_flag=init_flag,
                            bias_flag=bias_flag))
            self.input_channels = out_channels * block.expansion
            block_num -= 1
            self.exp_var += 1
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0(x)
        if self.pr_flag:
            print('stem out: chn #: {}, mean: {:0.3f}, std: {:0.3f}\n'.format(x.shape, float(torch.mean(torch.mean(x,dim=(0,2,3)))), float(torch.mean(torch.std(x,dim=(0,2,3)))) ) )
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        #print(x.shape)
        x = F.adaptive_avg_pool2d(x, 1) * math.sqrt(16)
        if self.pr_flag:
            print('pool out: chn #: {}, mean: {:0.3f}, std: {:0.3f}\n'.format(x.shape, float(torch.mean(torch.mean(x,dim=0))), float(torch.mean(torch.std(x,dim=0))) ) )
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        if self.pr_flag:
            print('linear out: chn #: {}, mean: {:0.3f}, std: {:0.3f}\n'.format(x.shape, float(torch.mean(torch.mean(x,dim=0))), float(torch.mean(torch.std(x,dim=0))) ) )
        return x

def resnet18(num_classes=100):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2],num_classes=num_classes)

def resnet34(num_classes=100):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3],num_classes=num_classes)

def resnet50(num_classes=100):
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3],num_classes=num_classes)

def resnet101(num_classes=100):
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3],num_classes=num_classes)

def resnet152(num_classes=100):
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])

def resnet(num_classes=100, depth=10, width=64):
    return ResNet(
            block=ResBlockv2,
            width=width,
            depth=depth,
            bn_flag=True,
            bias_flag=True,
            pr_flag=False,
            res_flag=True,
            init_flag=False,
            num_classes=num_classes
        )


"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, num_classes=100,embedding=False):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
        return output

def make_layers(cfg, batch_norm=False):
    #layers = []
    layers = nn.Sequential()
    input_channel = 3
    for i,l in enumerate(cfg):
        if l == 'M':
            layers.add_module('maxpool-{}'.format(i),nn.MaxPool2d(kernel_size=2, stride=2))
            continue

        layers.add_module('conv-{}'.format(i),nn.Conv2d(input_channel, l, kernel_size=3, padding=1,bias=False))

        if batch_norm:
            layers.add_module('bn-{}'.format(i),nn.BatchNorm2d(l))

        layers.add_module('relu-{}'.format(i),nn.ReLU(inplace=True))
        input_channel = l
    return layers

def vgg11(num_classes=100):
    return VGG(make_layers(cfg['A'], batch_norm=False),num_classes=num_classes)

def vgg13(num_classes=100):
    return VGG(make_layers(cfg['B'], batch_norm=False),num_classes=num_classes)

def vgg16(num_classes=100):
    return VGG(make_layers(cfg['D'], batch_norm=False),num_classes=num_classes)

def vgg19(num_classes=100):
    return VGG(make_layers(cfg['E'], batch_norm=False),num_classes=num_classes)


def vgg11_bn(num_classes=100):
    return VGG(make_layers(cfg['A'], batch_norm=True),num_classes=num_classes)

def vgg13_bn(num_classes=100):
    return VGG(make_layers(cfg['B'], batch_norm=True),num_classes=num_classes)

def vgg16_bn(num_classes=100):
    return VGG(make_layers(cfg['D'], batch_norm=True),num_classes=num_classes)

def vgg19_bn(num_classes=100):
    return VGG(make_layers(cfg['E'], batch_norm=True),num_classes=num_classes)
