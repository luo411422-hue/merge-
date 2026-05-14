import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
from models.ResNet import *
from  dataset import *
from pathlib import Path
# 导 入你的 ResNet 模型


""" with no pre_train"""

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

def evaluate(model, dataloader, criterion, device):
    model.eval()
    eval_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            eval_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    return eval_loss / len(dataloader), acc

def main():
    # 1. 基础设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 填入你本地的数据集路径
    data_root = r'E:\Python\datasets\CIFAR-100' 
    batch_size = 128
    epochs =200
    trainloader, valloader, testloader = CIFAR100.get_loader(
        root=data_root,
        batch_size=batch_size,
        test_batch_size=100,
        val_ratio=0.1,
        seed=42,
        num_workers=4,
    )
    
    # 4. 初始化模型 (Student: ResNet18)
    print("==> 初始化 ResNet-56 教师模型...")
    model = ResNet56(num_classes=100,dataset = "cifar100")
    model = model.to(device)
    ckpt_dir = Path(__file__).resolve().parent / "checkpoint"
    save_path = ckpt_dir / "Ter_resnet56-best.pth"
    best_acc = -1.0

    if save_path.exists():
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint["net"])
        best_acc = checkpoint.get("acc", best_acc)
        print(f"Loaded best weights from: {save_path.resolve()}")
    else:
        print("No existing checkpoint found, training from pretrained ImageNet weights.")

    # 5. 定义损失函数、优化器和学习率调度器
    criterion = nn.CrossEntropyLoss()
    # 标准的 CIFAR 训练超参数：SGD, lr=0.1, momentum=0.9, weight_decay=5e-4
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    # 在 60, 120, 160 个 epoch 时将学习率乘以 0.2
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2) #

    # 6. 开始训练循环
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} (LR: {scheduler.get_last_lr()[0]:.5f}) ---") 
        # 训练一轮
        train(epoch, model, trainloader, criterion, optimizer, device)
        # 验证一轮（只用验证集选最优模型）
        val_loss, val_acc = evaluate(model, valloader, criterion, device)
        print(f"==> Epoch {epoch} Val Results: Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        if val_acc > best_acc:
            print(f"🌟 验证准确率提升: {best_acc:.2f}% -> {val_acc:.2f}%! 正在保存模型...")
            os.makedirs('checkpoint', exist_ok=True)
            state = {
                'net': model.state_dict(),
                'acc': val_acc,
                'epoch': epoch,
            }
            torch.save(state, save_path)
            best_acc = val_acc
        
        # 更新学习率
        scheduler.step()

    # 最终只评估一次测试集
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt['net'])
    test_loss, test_acc = evaluate(model, testloader, criterion, device)

    print(f"\n✅ 训练完成！最佳验证准确率: {best_acc:.2f}%")
    print(f"Final Test Results: Loss: {test_loss:.4f} | Acc: {test_acc:.2f}%")
    print("权重已保存到:", save_path.resolve())

if __name__ == '__main__':
    main()