import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from ResNet import resnet18

if __name__ == '__main__':

    test_dataset = datasets.CIFAR10(root="../Datasets/CIFAR10/test", train=False, transform=transforms.ToTensor(),download=True)

    test_dataloader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=True, num_workers=0)

    network = resnet18()
    param = torch.load("resnet18_param_best.pkl")  # 加载模型
    network.load_state_dict(param)  # 将参数放入模型当中

    acc = []
    for data in test_dataloader:
        imgs, targets = data
        output = network(imgs)  # 输出预测
        _, pre_lab = torch.max(output, 1)  # 提取预测序列
        batch_acc = np.array(pre_lab == targets).sum() / test_dataloader.__len__()
        acc.append(batch_acc)

    acc = sum(acc) / len(acc)
    print("accuracy: ", acc)  # 输出正确率 0.9844999999999992
