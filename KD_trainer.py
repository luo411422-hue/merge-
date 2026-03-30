import os

import torch
import torch.nn.functional as F
import torch.optim as optim

import dataset
from models import ResNet20, ResNet56


def distill_loss(student_logits, teacher_logits, student_feat, teacher_feat, labels, alpha, beta, base_t):
    ce_loss = F.cross_entropy(student_logits, labels)

    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits, dim=1)
        entropy = -(teacher_probs * torch.log(teacher_probs + 1e-8)).sum(dim=1)
        max_entropy = torch.log(torch.tensor(teacher_logits.size(1), device=teacher_logits.device, dtype=teacher_logits.dtype))
        temperature = base_t * (1.0 + entropy / max_entropy.clamp_min(1e-8))
        temperature = temperature.unsqueeze(1)

    student_log_T = F.log_softmax(student_logits / temperature, dim=1)
    teacher_soft_T = F.softmax(teacher_logits / temperature, dim=1)
    kd_loss = F.kl_div(student_log_T, teacher_soft_T, reduction='none').sum(dim=1)
    kd_loss = (kd_loss * (temperature.squeeze(1) ** 2)).mean()

    feat_loss = F.mse_loss(student_feat, teacher_feat.detach())
    return (1.0 - alpha) * ce_loss + alpha * kd_loss + beta * feat_loss


def load_teacher(model, ckpt_path, device):
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=device)

    state_dict = checkpoint['net']
    model_has_module = next(iter(model.state_dict())).startswith('module.')
    ckpt_has_module = next(iter(state_dict)).startswith('module.')

    if model_has_module and not ckpt_has_module:
        state_dict = {'module.' + key: value for key, value in state_dict.items()}
    elif ckpt_has_module and not model_has_module:
        state_dict = {key[len('module.'):]: value for key, value in state_dict.items()}

    model.load_state_dict(state_dict)


def train(epoch, student, teacher, trainloader, optimizer, device, alpha, beta, base_t, epochs):
    student.train()
    teacher.eval()

    train_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            teacher_logits, teacher_feat = teacher(inputs, return_features=True, feature_layer='layer2')

        optimizer.zero_grad()
        student_logits, student_feat = student(inputs, return_features=True, feature_layer='layer2')
        loss = distill_loss(student_logits, teacher_logits, student_feat, teacher_feat, targets, alpha, beta, base_t)
        loss.backward() #损失反向传播
        optimizer.step()

        train_loss += loss.item()
        _, predicted = student_logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch [{batch_idx}/{len(trainloader)}] "
                      f"| Train Loss: {train_loss / (batch_idx + 1):.4f} "
                      f"| Train Acc: {100.0 * correct / total:.2f}%")
    # print(
    #     f"Epoch [{epoch}/{epochs}] "
    #     f"| Train Loss: {train_loss / len(trainloader):.4f} "
    #     f"| Train Acc: {100.0 * correct / total:.2f}%"
    # )


def test(epoch, student, testloader, device, best_acc, save_name):
    student.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = student(inputs)
            loss = F.cross_entropy(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            

    acc = 100.0 * correct / total
    print(f"==> Test Loss: {test_loss / len(testloader):.4f} | Test Acc: {acc:.2f}%")

    if acc > best_acc:
        print(f"==> Saving model: {best_acc:.2f}% -> {acc:.2f}%")
        model_to_save = student.module if hasattr(student, 'module') else student
        os.makedirs('checkpoint', exist_ok=True)
        torch.save(
            {'net': model_to_save.state_dict(), 'acc': acc, 'epoch': epoch},
            os.path.join('checkpoint', save_name)
        )
        best_acc = acc

    return best_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = r'E:\Python\datasets\CIFAR-100'
   # teacher_ckpt = os.path.join(project_dir, 'checkpoint', 'teacher_resnet56_best.pth')
    teacher_ckpt = os.path.join(
    r"E:\paper\feature\checkpoint",
    "teacher_resnet56_best.pth"
    )
    print(f"Teacher checkpoint path: {teacher_ckpt}")
    student_ckpt = 'student_resnet20_adaptive_layer2.pth'

    batch_size = 128
    epochs = 300
    lr = 0.1
    alpha = 0.9
    beta = 0.3
    base_t = 4.0

    print('==> Preparing CIFAR100 data..')
    trainloader, testloader = dataset.get_loader(
        root=data_root,
        batch_size=batch_size,
        test_batch_size=100,
        num_workers=2
    )

    print('==> Building teacher and student..')
    teacher = ResNet56(dataset='cifar100', num_classes=100).to(device)
    student = ResNet20(dataset='cifar100', num_classes=100).to(device)

    if torch.cuda.is_available():
        teacher = torch.nn.DataParallel(teacher)
        student = torch.nn.DataParallel(student)

    if not os.path.isfile(teacher_ckpt):
        raise FileNotFoundError(f'Teacher checkpoint not found: {teacher_ckpt}')
    if os.path.getsize(teacher_ckpt) == 0:
        raise RuntimeError(f'Teacher checkpoint is empty: {teacher_ckpt}')

    print(f'==> Loading teacher checkpoint from {teacher_ckpt}..')
    load_teacher(teacher, teacher_ckpt, device)

    optimizer = optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train(epoch, student, teacher, trainloader, optimizer, device, alpha, beta, base_t, epochs)
        best_acc = test(epoch, student, testloader, device, best_acc, student_ckpt)
        scheduler.step()

    print(f'Training finished. Best Acc: {best_acc:.2f}%')


if __name__ == '__main__':
    main()
