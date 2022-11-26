import torch.nn as nn
import torch

cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_features(cfg: list):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)

feature=make_features(cfgs["vgg16"])


class VGG(nn.Module):
    def __init__(self, num_classes=1000):    #在这里放入其他参数无非是在下面直接调用这个参数，也可以不写这些参数直接在下边给出值就行
        super(VGG, self).__init__()
        self.features = feature
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)  #从第一个维度开始展平，因为第一个是batch
        # N x 512*7*7
        x = self.classifier(x)
        return x

if __name__=="__main__":
    print(VGG())







'''
1.网络模型就是一个类，一个框架，就是3部分 1.导入库 2.继承基模型框架然后定义自己网络所需要的神经网络的每一层【如定义卷积层，池化层，全连接层，激活函数】 3.前向传播 
'''