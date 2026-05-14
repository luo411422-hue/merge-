import matplotlib.pyplot as plt

# 1. 准备数据 (基于图二 ResNet56 -> ResNet20 列)
methods = ['Student Alone', 'Standard KD', 'KD + AT', 'KD + Adaptive T', 'Ours']
accuracy = [69.8, 70.66, 71.40, 71.90, 72.1]

# 计算相对增益 (相对于 Student Alone 的提升)
baseline = accuracy[0]
gains = [acc - baseline for acc in accuracy]

# 2. 创建画布
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=100)

# 3. 绘制准确率曲线 (左轴 - 蓝色)
line1 = ax1.plot(methods, accuracy, marker='o', color='#1f77b4', linewidth=2, markersize=8, label='Accuracy (%)')
ax1.set_ylabel('Accuracy (%)', color='#1f77b4', fontsize=12)
ax1.tick_params(axis='y', labelcolor='#1f77b4')
ax1.set_ylim(67, 74)  # 根据数据调整范围
ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

# 在点上标注具体数值
for i, txt in enumerate(accuracy):
    ax1.annotate(f'{txt}%', (methods[i], accuracy[i]), textcoords="offset points", 
                 xytext=(0, 10), ha='center', color='#1f77b4', fontweight='bold')

# 4. 绘制增益曲线 (右轴 - 橙色)
ax2 = ax1.twinx()
line2 = ax2.plot(methods, gains, marker='s', color='#ff7f0e', linestyle='--', linewidth=2, markersize=8, label='Relative Gain (%)')
ax2.set_ylabel('Relative Gain (%)', color='#ff7f0e', fontsize=12)
ax2.tick_params(axis='y', labelcolor='#ff7f0e')
ax2.set_ylim(0, 4) # 增益范围

# 在点上标注增益数值 (带方框样式)
for i, txt in enumerate(gains):
    if i == 0: continue # 跳过第一个点(0)
    ax2.annotate(f'+{txt:.2f}%', (methods[i], gains[i]), textcoords="offset points", 
                 xytext=(0, -20), ha='center', color='#ff7f0e', 
                 bbox=dict(boxstyle='round,pad=0.3', edgecolor='#ff7f0e', facecolor='white'))

# 5. 图表修饰
plt.title('$Performance Comparison: ResNet56 to ResNet20 (CIFAR-100)$', fontsize=14, pad=20)
ax1.set_xlabel('Methods', fontsize=12)

# 合并图例
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

plt.tight_layout()
plt.show()