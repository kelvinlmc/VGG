import torch.nn as nn



cfg=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
layers = []
in_channels = 3
for v in cfg:
    print(layers)
    if v == "M":
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    else:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
        layers += [conv2d, nn.ReLU(True)]
        in_channels = v
print(nn.Sequential(*layers))
'''
cfg=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
layers = []
for v in cfg:
    print(layers)
    if v == "M":
        layers +=[v]
    else:
        layers += [v]
        in_channels = v

'''