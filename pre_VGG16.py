import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import VGG16_Weights, vgg16

from dataset import get_cifar100_imagenet_style_loader

os.environ["TORCH_HOME"] = r"E:\paper\transformer-master\weight"


def train(epoch, epochs, model, trainloader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

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

    return train_loss / len(trainloader), 100.0 * correct / total

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    eval_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        eval_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100.0 * correct / total
    return eval_loss / len(dataloader), acc


def build_model(num_classes, freeze_backbone):
    weights = VGG16_Weights.IMAGENET1K_V1
    model = vgg16(weights=weights)

    # Keep the pretrained feature extractor intact and only replace the
    # classification output for CIFAR-100.
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    return model


def build_optimizer(model, freeze_backbone):
    if freeze_backbone:
        return optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3,
            momentum=0.9,
            weight_decay=5e-4,
        )

    return optim.SGD(
        [
            {"params": model.features.parameters(), "lr": 1e-4},
            {"params": model.classifier.parameters(), "lr": 1e-3},
        ],
        momentum=0.9,
        weight_decay=5e-4,
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    data_root = r"E:\Python\datasets\CIFAR-100"
    batch_size = 64
    epochs = 100
    num_classes = 100
    freeze_backbone = False

    # Pretrained ImageNet weights expect ImageNet-style preprocessing, so we
    # resize CIFAR-100 from 32x32 to 224x224 and use ImageNet normalization.
    trainloader, valloader, testloader = get_cifar100_imagenet_style_loader(
        root=data_root,
        batch_size=batch_size,
        test_batch_size=100,
        val_ratio=0.1,
        seed=42,
        num_workers=2,
    )

    print("==> Initializing pretrained VGG16...")
    print(f"Freeze backbone: {freeze_backbone}")
    model = build_model(num_classes=num_classes, freeze_backbone=freeze_backbone).to(device)

    ckpt_dir = Path(__file__).resolve().parent / "checkpoint"
    save_path = ckpt_dir / "Pre_VGG16-best.pth"
    best_acc = -1.0

    if save_path.exists():
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint["net"])
        best_acc = checkpoint.get("acc", best_acc)
        print(f"Loaded best weights from: {save_path.resolve()}")
    else:
        print("No existing checkpoint found, training from pretrained ImageNet weights.")

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, freeze_backbone)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 70, 90], gamma=0.2)

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n--- Epoch {epoch}/{epochs} (LR: {current_lr:.6f}) ---")
        train(epoch, epochs, model, trainloader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, valloader, criterion, device)
        print(f"==> Epoch {epoch} Val Results: Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            print(f"Validation accuracy improved: {best_acc:.2f}% -> {val_acc:.2f}%. Saving model...")
            os.makedirs(ckpt_dir, exist_ok=True)
            state = {
                "net": model.state_dict(),
                "acc": val_acc,
                "epoch": epoch,
                "freeze_backbone": freeze_backbone,
            }
            torch.save(state, save_path)
            best_acc = val_acc

        scheduler.step()

    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["net"])
    test_loss, test_acc = evaluate(model, testloader, criterion, device)

    print(f"\nTraining finished. Best val acc: {best_acc:.2f}%")
    print(f"Final Test Results: Loss: {test_loss:.4f} | Acc: {test_acc:.2f}%")
    print(f"Best checkpoint saved to: {save_path.resolve()}")


if __name__ == "__main__":
    main()
