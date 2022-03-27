import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from AlexNet import AlexNet

if __name__ == '__main__':

    test_data = CIFAR10("../Datasets/CIFAR10/test", train=False, transform=transforms.ToTensor(), download=False)

    test_dataloader = DataLoader(dataset=test_data, batch_size=100, shuffle=True, num_workers=0)

    network = AlexNet()
    model = torch.load("AlexNet_param_best.pkl")  # 加载模型
    network.load_state_dict(model)  # 将参数放入模型当中

    acc = []
    for data in test_dataloader:
        imgs, targets = data
        output = network(imgs)  # 输出预测
        _, pre_lab = torch.max(output, 1)  # 提取预测序列
        batch_acc = np.array(pre_lab == targets).sum() / test_dataloader.__len__()
        acc.append(batch_acc)

    acc = sum(acc) / len(acc)
    print("accuracy: ", acc)  # 输出正确率
