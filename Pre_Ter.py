import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet50 ,ResNet50_Weights
import dataset

import os
os.environ["TORCH_HOME"] = r"E:\paper\transformer-master\weight"

def train(epoch, epochs, model, trainloader, criterion, optimizer, device):
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

        if batch_idx % 20 == 0:
            print(
                f"Epoch [{epoch}/{epochs}] Batch [{batch_idx + 1}/{len(trainloader)}] "
                f"| Train Loss: {train_loss / (batch_idx + 1):.4f} "
                f"| Train Acc: {100.0 * correct / total:.2f}%"
            )


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

    acc = 100.0 * correct / total
    return eval_loss / len(dataloader), acc


def get_cifar100_imagenet_style_loader(
    root=r'E:\Python\datasets\CIFAR-100',
    batch_size=128,
    test_batch_size=100,
    val_ratio=0.1,
    seed=42,
    num_workers=2,
):
    """使CIFAR100的图像尺寸与imagenet图片尺寸对齐"""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    transform_eval = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    base_trainset = torchvision.datasets.CIFAR100(
        root=root, train=True, download=True, transform=None
    )
    total_train = len(base_trainset)
    val_size = int(total_train * val_ratio)
    train_size = total_train - val_size
    if val_size <= 0 or train_size <= 0:
        raise ValueError("val_ratio creates empty split; please use a value in (0, 1)")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_train, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    trainset_aug = torchvision.datasets.CIFAR100(
        root=root, train=True, download=False, transform=transform_train
    )
    trainset_eval = torchvision.datasets.CIFAR100(
        root=root, train=True, download=False, transform=transform_eval
    )
    testset = torchvision.datasets.CIFAR100(
        root=root, train=False, download=True, transform=transform_eval
    )

    trainset = Subset(trainset_aug, train_indices)
    valset = Subset(trainset_eval, val_indices)
    pin = torch.cuda.is_available()
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin
    )
    valloader = DataLoader(
        valset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin
    )
    testloader = DataLoader(
        testset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin
    )
    return trainloader, valloader, testloader



    


def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    freeze_backbone = True
    data_root = r"E:\Python\datasets\CIFAR-100"
    batch_size = 128
    epochs = 200
    num_classes = 100

    trainloader, valloader, testloader = get_cifar100_imagenet_style_loader(
        root=data_root,
        batch_size=batch_size,
        test_batch_size=100,
        val_ratio=0.1,
        seed=42,
        num_workers=2,
    )

    print("==> 初始化预训练 ResNet50 教师模型...")
   
    try:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    except Exception:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc"):    #只对全连接层进行参数更新，backbone无需参数更新
                param.requires_grad = False
    model = model.to(device)

    save_path = Path(__file__).resolve().parent / "weight"/ "Pre_ResNet50-best.pth"
    
    
    if save_path.exists():
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint["net"])
        print(f"Loaded best weights from: {save_path.resolve()}")
    else:
        print("No existing checkpoint found, training from initialized weights.")
        
    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(trainable_params, lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)


    best_acc = -1.0



    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} (LR: {scheduler.get_last_lr()[0]:.5f}) ---")
        train(epoch, epochs, model, trainloader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, valloader, criterion, device)
        print(f"==> Epoch {epoch} Val Results: Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            print(f"验证准确率提升: {best_acc:.2f}% -> {val_acc:.2f}%! 正在保存模型...")
            os.makedirs(save_path, exist_ok=True)
            state = {
                "net": model.state_dict(),
                "acc": val_acc,
                "epoch": epoch,
            }
            torch.save(state, save_path)
            best_acc = val_acc

        scheduler.step()

    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt["net"])
    test_loss, test_acc = evaluate(model, testloader, criterion, device)

    print(f"\n训练完成！最佳验证准确率: {best_acc:.2f}%")
    print(f"Final Test Results: Loss: {test_loss:.4f} | Acc: {test_acc:.2f}%")
    print(f"权重已保存至: {save_path}")


if __name__ == "__main__":
    main()
