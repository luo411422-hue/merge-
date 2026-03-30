import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
from models import *
import dataset
# 导入你的 ResNet 模型


def train(epoch, model, trainloader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 每 100 个 batch 打印一次进度
        if batch_idx % 20 == 0:
            print(f"Epoch [{epoch}/{200}] Batch [{batch_idx+1}/{len(trainloader)}] "
                  f"| Train Loss: {train_loss/(batch_idx+1):.4f} "
                  f"| Train Acc: {100.*correct/total:.2f}%")

def test(epoch, model, testloader, criterion, device, best_acc):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    print(f"==> Epoch {epoch} Test Results: Loss: {test_loss/len(testloader):.4f} | Acc: {acc:.2f}%")

    # 保存表现最好的模型
    if acc > best_acc:
        print(f"🌟 测试准确率提升: {best_acc:.2f}% -> {acc:.2f}%! 正在保存模型...")
        # 确保存在 checkpoints 文件夹
        if not os.path.exists('checkpoint'):
            os.makedirs('checkpoint')
        
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, './checkpoint/teacher_resnet56_best.pth')
        best_acc = acc

    return best_acc

def main():
    # 1. 基础设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 填入你本地的数据集路径
    data_root = r'E:\Python\datasets\CIFAR-100' 
    batch_size = 128
    epochs = 200
    trainloader, testloader = dataset.get_loader(root=data_root, batch_size=batch_size)
    
    # 4. 初始化模型 (Student: ResNet20)
    print("==> 初始化 ResNet-56 学生模型...")
    model = ResNet56(num_classes=100)
    model = model.to(device)

    # 5. 定义损失函数、优化器和学习率调度器
    criterion = nn.CrossEntropyLoss()
    # 标准的 CIFAR 训练超参数：SGD, lr=0.1, momentum=0.9, weight_decay=5e-4
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    # 在 60, 120, 160 个 epoch 时将学习率乘以 0.2
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    # 6. 开始训练循环
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} (LR: {scheduler.get_last_lr()[0]:.5f}) ---")
        
        # 训练一轮
        train(epoch, model, trainloader, criterion, optimizer, device)
        
        # 测试一轮
        best_acc = test(epoch, model, testloader, criterion, device, best_acc)
        
        # 更新学习率
        scheduler.step()

    print(f"\n✅ 训练完成！教师模型最高准确率为: {best_acc:.2f}%")
    print("权重已保存至: ./checkpoint/teacher_resnet56_best.pth")

if __name__ == '__main__':
    main()