# MyClassicNet
神经网络合集

## LeNet
### 网络结构
```python
import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)), # torch.Size([1, 6, 24, 24])
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  # torch.Size([1, 6, 12, 12])
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1)),  # torch.Size([1, 16, 8, 8])
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  # torch.Size([1, 16, 4, 4])
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),  # torch.Size([1, 256])
            nn.Linear(in_features=16 * 5 * 5, out_features=120),  # torch.Size([1, 120])
            nn.Sigmoid(),
            nn.Linear(in_features=120, out_features=84),  # torch.Size([1, 84])
            nn.Sigmoid(),
            nn.Linear(in_features=84, out_features=10)  # torch.Size([1, 10])
        )

    def forward(self, x):
        x = self.conv1(x) # torch.Size([1, 6, 14, 14])
        x = self.conv2(x) # torch.Size([1, 16, 5, 5])
        x = self.fc(x) # torch.Size([1, 10])
        return x
```
### 运行结果

- 数据集 MNIST

- epoch=50 
- batch_size=64 
- 显卡 NVIDA 940MX 
- 优化器 Adam 
- 损失函数 CrossEntropyLoss

![image-20220327092300583](https://raw.githubusercontent.com/dongchao0612/image/main/image-20220327092300583.png)

## AlexNet

### 网络结构
```python
import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(3, 3), stride=(1, 1)),  # torch.Size([64, 96, 30, 30]
            nn.BatchNorm2d(96),  # torch.Size([64, 96, 30, 30])
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)  # torch.Size([64, 96, 14, 14])
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=(3, 3), stride=(1, 1)),  # torch.Size([64, 256, 12, 12])
            nn.BatchNorm2d(256),  # torch.Size([64, 256, 12, 12])
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2)  # torch.Size([64, 256, 5, 5])
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=1),  # torch.Size([64, 384, 5, 5])
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=1),  # torch.Size([64, 384, 5, 5])
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),  # torch.Size([64, 256, 5, 5])
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2)  # torch.Size([64, 256, 2, 2])
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),  # torch.Size([64, 1024])
            nn.Linear(256 * 2 * 2, 2048),  # torch.Size([64, 2048])
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # torch.Size([64, 2048])
            nn.Linear(2048, 2048),  # torch.Size([64, 2048])
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # torch.Size([64, 2048])
            nn.Linear(2048, 10)  # torch.Size([64, 10])
        )

    def forward(self, x):
        x = self.conv1(x)  # torch.Size([64, 96, 14, 14])
        x = self.conv2(x)  # torch.Size([64, 256, 5, 5])
        x = self.conv3(x)  # torch.Size([64, 256, 2, 2])
        x = self.fc(x)  # torch.Size([64, 10])
        return x
```
### 运行结果

- 数据集 CIFAR10

- epoch=500 
- batch_size=64 
- 显卡 NVIDA 940MX 
- 优化器 Adam 
- 损失函数 CrossEntropyLoss

![image-20220327185512662](https://raw.githubusercontent.com/dongchao0612/image/main/image-20220327185512662.png)

## GoogLeNet
### 网络结构
```python
import torch
from torch import nn


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding), nn.ReLU(inplace=True))


class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(Inception, self).__init__()
        self.branch1 = conv(in_channels, c1, kernel_size=1)
        self.branch2 = nn.Sequential(
            conv(in_channels, c2[0], kernel_size=1),
            conv(c2[0], c2[1], kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(
            conv(in_channels, c3[0], kernel_size=1),
            conv(c3[0], c3[1], kernel_size=5, stride=1, padding=2))
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv(in_channels, c4, kernel_size=1))

    def forward(self, x):
        block1 = self.branch1(x)
        block2 = self.branch2(x)
        block3 = self.branch3(x)
        block4 = self.branch4(x)

        block = (block1, block2, block3, block4)

        return torch.cat(block, dim=1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.block1 = nn.Sequential(
            conv(3, 64, kernel_size=(7, 7), stride=2, padding=3),  # torch.Size([64, 64, 16, 16])
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # torch.Size([64, 64, 8, 8])
        )
        self.block2 = nn.Sequential(
            conv(64, 64, kernel_size=1),  # torch.Size([64, 64, 8, 8])
            conv(64, 192, kernel_size=3, stride=1, padding=1),  # torch.Size([64, 192, 8, 8])
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # torch.Size([64, 192, 4, 4])
        )
        self.block3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),  # torch.Size([64, 256, 4, 4])
            Inception(256, 128, (128, 192), (32, 96), 64),  # torch.Size([64, 480, 4, 4])
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # torch.Size([64, 480, 2, 2])
        )
        self.block4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),  # torch.Size([64, 512, 2, 2])
            Inception(512, 160, (112, 224), (24, 64), 64),  # torch.Size([64, 512, 2, 2])
            Inception(512, 128, (128, 256), (24, 64), 64),  # torch.Size([64, 512, 2, 2])
            Inception(512, 112, (144, 288), (32, 64), 64),  # torch.Size([64, 528, 2, 2])
            Inception(528, 256, (160, 320), (32, 128), 128),  # torch.Size([64, 832, 2, 2])
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # torch.Size([64, 832, 1, 1])
        )
        self.block5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),  # torch.Size([64, 832, 1, 1])
            Inception(832, 384, (192, 384), (48, 128), 128),  # torch.Size([64, 1024, 1, 1])
            nn.AdaptiveAvgPool2d((1, 1)),  # torch.Size([64, 1024, 1, 1])
            nn.Dropout(0.4),  # torch.Size([64, 1024, 1, 1])
            nn.Flatten()  # torch.Size([64, 1024])
        )
        self.classifier = nn.Linear(1024, 10)  # torch.Size([64, 10])

    def forward(self, x):
        x = self.block1(x)  # torch.Size([64, 64, 8, 8])
        x = self.block2(x)  # torch.Size([64, 192, 4, 4])
        x = self.block3(x)  # torch.Size([64, 480, 2, 2])
        x = self.block4(x)  # torch.Size([64, 832, 1, 1])
        x = self.block5(x)  # torch.Size([64, 1024])
        x = self.classifier(x)  # torch.Size([64, 10])
        return x



```
### 运行结果

- 数据集 CIFAR10
- epoch=500 
- batch_size=64 
- 显卡 NVIDA 940MX 
- 优化器 Adam 
- 损失函数 CrossEntropyLoss

![image-20220327185315276](https://raw.githubusercontent.com/dongchao0612/image/main/image-20220327185315276.png)