import sys
#sys.path.append("..")
import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

from VggNet import VGGNet11



if __name__ == '__main__':

    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    # 训练数据
    train_dataset = datasets.CIFAR10(root="../Datasets/CIFAR10/train", train=True, transform=transform_train, download=True)
    test_dataset = datasets.CIFAR10(root="../Datasets/CIFAR10/test", train=False, transform=transform_test, download=True)
    print(f"训练datasets的长度：{train_dataset.__len__()}，测试datasets的长度：{test_dataset.__len__()}")  # 50000 10000
    # 数据加载器
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True, num_workers=0)
    print(f"训练DataLoader的长度：{train_dataloader.__len__()}，测试DataLoader的长度：{test_dataloader.__len__()}")

    # 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备：", device)

    # 定义网络
    torch.manual_seed(13)
    model = VGGNet11().to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fun = nn.CrossEntropyLoss()
    loss_fun.to(device)

    epoch = 5
    total_train_step = 0  # 记录训练次数
    total_test_step = 0  # 记录测试次数
    best_acc = 0

    writer=SummaryWriter("VggNetlogs")
    # input_data = torch.randn((64, 3, 28, 28))
    # input_data = input_data.to(device)
    # writer.add_graph(model, input_data)
    # writer.close()

    for e in range(epoch):
        print("================= EPOCH: {}/{} ===============".format(e + 1, epoch))
        # Train
        model.train()
        total_train_loss = 0
        for data in train_dataloader:
            imgs, target = data
            #print(imgs.shape, target) #torch.Size([64, 3, 32, 32])
            imgs = imgs.to(device)
            target = target.to(device)
            output = model(imgs)
            loss = loss_fun(output, target)  # 损失值

            # 优化器优化模型
            optimizer.zero_grad()   # 优化器梯度清零
            loss.backward()         # 反向传播
            optimizer.step()        # 优化

            total_train_loss = total_train_loss + loss.item()
            total_train_step += 1

        #     if total_train_step % 100 == 0:
        #        print(f"训练次数 {total_train_step} ， 测试损失值 {loss.item()}")

        print(f"整体训练集的Loss：{total_train_loss}")

        # 测试步骤开始
        model.eval()
        total_test_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for data in test_dataloader:
                imgs, target = data
                imgs = imgs.to(device)
                target = target.to(device)
                outputs = model(imgs)
                loss = loss_fun(outputs, target)
                total_test_loss = total_test_loss + loss.item()
                accuacy = (outputs.argmax(1) == target).sum()
                total_accuracy += accuacy
                total_test_step += 1

                # if total_test_step % 10 == 0:
                #     print(f"测试次数 {total_test_step} ， 测试损失值 {loss.item()}")

        Accuacy=(total_accuracy / test_dataset.__len__()).item() #<class 'torch.Tensor'>
        writer.add_scalar(tag="accuracy", scalar_value=Accuacy,global_step=e)
        #print("整体测试集的Accuacy：",Accuacy)
        if Accuacy > best_acc:
            best_acc = Accuacy
            print("整体测试集的best Accuacy",best_acc)
            torch.save(model.state_dict(), "VGGNet11_param_best.pkl")

    writer.close()
    print("训练结束，模型已保存")
