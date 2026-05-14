import os

import torch
import torch.nn.functional as F
import torch.optim as optim

from dataset import *
from models.ResNet import ResNet110, ResNet32
from pathlib import Path

def _attention_map(feat):
    # Attention transfer: sum of squared activations over channels.
    att = feat.pow(2).mean(dim=1, keepdim=True)
    return F.normalize(att.flatten(1), p=2, dim=1)


def _feature_distill_loss(student_feat, teacher_feat):
    teacher_feat = teacher_feat.detach()
    if student_feat.shape[-2:] != teacher_feat.shape[-2:]:
        student_feat = F.adaptive_avg_pool2d(student_feat, teacher_feat.shape[-2:])
    return F.mse_loss(_attention_map(student_feat), _attention_map(teacher_feat))


def distill_loss(student_logits, teacher_logits, student_feat, teacher_feat, labels, alpha, beta, base_t):
    """ 自适应温度 + 中间层匹配"""
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

    feat_loss = _feature_distill_loss(student_feat, teacher_feat)
    return (1.0 - alpha) * ce_loss + alpha * kd_loss + beta * feat_loss

def dis_loss(student_logits, teacher_logits,labels,alpha,base_t):
    """ 普通蒸馏"""
    ce_loss =  F.cross_entropy(student_logits, labels)
    
    with torch.no_grad():
        temperature = torch.full(
            (student_logits.size(0), 1),
            fill_value=base_t,
            device=student_logits.device,
            dtype=student_logits.dtype
        )
    student_log_T = F.log_softmax(student_logits / temperature, dim=1)
    teacher_soft_T = F.softmax(teacher_logits / temperature, dim=1)  
    kd_loss = F.kl_div(student_log_T, teacher_soft_T, reduction='none').sum(dim=1)
    kd_loss = (kd_loss * (temperature.squeeze(1) ** 2)).mean()

    return (1.0 - alpha) * ce_loss + alpha * kd_loss

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


def train(epoch, student, teacher, trainloader, optimizer, device, alpha, beta, base_t, epochs,normal_KD):
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
        if normal_KD:
            loss = dis_loss(student_logits, teacher_logits, targets, alpha, base_t)
            
        else:
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


def evaluate(student, dataloader, device):
    student.eval()
    eval_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = student(inputs)
            loss = F.cross_entropy(outputs, targets)

            eval_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.0 * correct / total
    return eval_loss / len(dataloader), acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = r'E:\Python\datasets\CIFAR-100'
   # teacher_ckpt = os.path.join(project_dir, 'checkpoint', 'teacher_resnet56_best.pth')
    teacher_ckpt = Path(__file__).resolve().parent / 'checkpoint' / 'Ter_resnet110-best.pth'
    print(f"Teacher checkpoint path: {teacher_ckpt}")
    student_ckpt = Path(__file__).resolve().parent / 'checkpoint' / 'student_resnet32_adaptive_layer2.pth'

    batch_size = 128
    epochs = 300
    lr = 0.1
    alpha = 0.1
    beta = 0.3
    base_t = 4.0
    normal_KD  = False
    print('==> Preparing CIFAR100 data..')
    trainloader, valloader, testloader = CIFAR100.get_loader(
        root=data_root,
        batch_size=batch_size,
        test_batch_size=100,
        val_ratio=0.1,
        seed=42,
        num_workers=2
    )

    print('==> Building teacher and student..')
    teacher = ResNet110(dataset='cifar100', num_classes=100).to(device)
    student = ResNet32(dataset='cifar100', num_classes=100).to(device)

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
        train(epoch, student, teacher, trainloader, optimizer, device, alpha, beta, base_t, epochs,normal_KD=False)
        val_loss, val_acc = evaluate(student, valloader, device)
        print(f"==> Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        if val_acc > best_acc:
            print(f"==> Saving model: {best_acc:.2f}% -> {val_acc:.2f}%")
            model_to_save = student.module if hasattr(student, 'module') else student
            os.makedirs('checkpoint', exist_ok=True)
            torch.save(
                {'net': model_to_save.state_dict(), 'acc': val_acc, 'epoch': epoch},
                student_ckpt
            )
            best_acc = val_acc
        scheduler.step()

    best_model_path =  student_ckpt
    checkpoint = torch.load(best_model_path, map_location=device)
    model_to_load = student.module if hasattr(student, 'module') else student
    model_to_load.load_state_dict(checkpoint['net'])
    test_loss, test_acc = evaluate(student, testloader, device)

    print(f'Training finished. Best Val Acc: {best_acc:.2f}%')
    print(f'Final Test Loss: {test_loss:.4f} | Final Test Acc: {test_acc:.2f}%')


if __name__ == '__main__':
    main()
