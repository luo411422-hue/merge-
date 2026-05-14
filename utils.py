'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init
import torch

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

import shutil

try:
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)
except Exception:
    # Fallback for Windows or when terminal size cannot be read
    try:
        term_width = shutil.get_terminal_size((80, 20)).columns
    except Exception:
        term_width = 80

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
            f += str(secondsf) + 's'
            i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f



import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体（可选，避免中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据
models = ['Normal ViT', 'KD-ViT (Logits)', 'DeiT Distillation']
accuracies = [83.9, 85.3, 88.4]
improvements = ['', '+1.4%', '+4.5%']  # 提升标注

# 颜色（柔和但对比鲜明）
colors = ['#6c91b2', '#e5a36f', '#7fbc6c']

# 设置画布大小和分辨率
fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

# 绘制柱状图，添加阴影效果
bars = ax.bar(models, accuracies, width=0.6, color=colors, 
              edgecolor='white', linewidth=1.2, 
              alpha=0.85, zorder=2)

# 在柱顶添加数值标签和提升标注
for bar, acc, imp in zip(bars, accuracies, improvements):
    height = bar.get_height()
    # 数值标签
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
            f'{acc}%', ha='center', va='bottom', 
            fontsize=14, fontweight='bold', color='#2c3e50')
    # 提升标注（如果不是基准模型）
    if imp:
        ax.text(bar.get_x() + bar.get_width()/2., height - 1.2,
                imp, ha='center', va='top', 
                fontsize=11, color='#d9534f', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

# 添加水平参考线（平均值虚线）
ax.axhline(y=accuracies[0], color='gray', linestyle='--', linewidth=1, alpha=0.6, zorder=1)
ax.text(len(models)-0.8, accuracies[0] + 0.2, f'Baseline: {accuracies[0]}%', 
        fontsize=9, color='gray', ha='right')

# 设置坐标轴
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='semibold')
ax.set_title('Model Performance Comparison on Cassava Leaf Disease', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(82, 90)  # 适当留白
ax.set_xlabel('Model', fontsize=13, fontweight='semibold')

# 网格样式（仅水平线）
ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
ax.set_axisbelow(True)

# 去除顶部和右侧边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

# 调整刻度标签字体
ax.tick_params(axis='x', labelsize=12, labelrotation=15)
ax.tick_params(axis='y', labelsize=11)

# 添加简要说明（可选）
fig.text(0.5, 0.01, 'DeiT distillation achieves the best performance (+4.5% over Normal ViT)', 
         ha='center', fontsize=10, color='#555555')

plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # 为底部文字留空间
plt.savefig('model_accuracy_comparison_optimized.png', dpi=300, bbox_inches='tight')
plt.show()