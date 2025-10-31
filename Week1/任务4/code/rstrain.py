import torch
import torchvision
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import os

pre_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  # 随机裁剪并缩放成224x224
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(
    root = "C:\\Users\\17765\\mmpretrain\\projects\\RoboM\\DataSet\\train",            # 训练集路径
    transform = pre_transform,                 # 应用预处理
)

val_dataset = datasets.ImageFolder(
    root = "C:\\Users\\17765\\mmpretrain\\projects\\RoboM\\DataSet\\val",            # 训练集路径
    transform = pre_transform,
)
# test_dataset = datasets.ImageFolder(
#     root="./DataKaggle_restructured/test1",              # 测试集路径
#     transform=pre_transform
# )
# 假设原始训练集为 train_dataset
# train_size = int(0.8 * len(train_dataset))  # 80% 训练
# val_size = len(train_dataset) - train_size  # 20% 验证

# train_subset, val_subset = random_split(
#     train_dataset,
#     [train_size, val_size],
#     generator=torch.Generator().manual_seed(42)  # 固定随机种子
# )

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last = True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last = True)
# test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False, drop_last = False)
writer = SummaryWriter("logs_herb")

# Squeeze & Excitation block
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channel, channel // reduction)
        self.fc2 = nn.Linear(channel // reduction, channel)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [batch_size, channel]
        y = x.mean(dim=0, keepdim=True) if x.dim() == 1 else x  # 如果是一维向量，保持维度
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        res18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.se = SEBlock(channel=1000, reduction=16)
        self.classifier2 = torch.nn.Linear(1000, 8)

        self.res18 = res18

    def forward(self, x):
        x = self.res18(x)
        x = self.se(x)  # 增加SE注意力
        x = self.classifier2(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model2 = MyModel().to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.SGD([
    {'params': model2.res18.layer1.parameters(), 'lr': 0.005},
    {'params': model2.res18.layer2.parameters(), 'lr': 0.005},
    {'params': model2.res18.layer3.parameters(), 'lr': 0.01},
    {'params': model2.res18.layer4.parameters(), 'lr': 0.01},
    {'params': model2.res18.fc.parameters(), 'lr': 0.02},
    {'params': model2.se.parameters(), 'lr': 0.01},
    {'params': model2.classifier2.parameters(), 'lr': 0.02}
], momentum=0.9, weight_decay=2e-5)
# optimizer = torch.optim.AdamW([
#     {'params': model2.res18.parameters(), 'lr': 5e-3},      # backbone 用较小学习率
#     {'params': model2.se.parameters(), 'lr': 5e-3},
#     {'params': model2.classifier2.parameters(), 'lr': 1e-2}     # 分类头用较大学习率
# ], weight_decay = 1e-4)

scheduler = {
    torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.2),
    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0),
    torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=30)
    }
# # 检验是否成功读取上次运行的参数
# checkpoint_path = 'res18_backbone.pth'
# if os.path.exists(checkpoint_path):
#     # miss检验点
#     checkpoint = torch.load('res18_backbone.pth')
#     missing, unexpected = model2.load_state_dict(checkpoint['model'], strict=False)
#     print("missing keys:", missing)
#     print("unexpected keys:", unexpected)
#
#     checkpoint = torch.load(checkpoint_path)
#
#     optimizer.load_state_dict(checkpoint['optimizer'])
#     start_epoch = checkpoint['epoch'] + 1  # 从下一个epoch开始训练
#     print(f"Loaded checkpoint. Resuming from epoch {start_epoch}")
# else:
#     start_epoch = 0
#     print("No checkpoint found. Training from scratch.")

# 定义早停参数
best_val_loss = float('inf')
patience = 2  # 容忍的连续负增长轮数
patience_counter = 0
best_epoch = 0

for epoch in range(20):
    # 数据集
    model2.train()
    running_loss = 0.0
    if(epoch == 0):
        lr = 5e-2
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model2(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 每200个batch记录一次
        if i % 200 == 199:
            writer.add_scalar('training loss', running_loss / 200, epoch * len(train_loader) + i)
            print(f'Epoch [{epoch + 1}/20], Batch [{i + 1}/{len(train_loader)}], Loss: {running_loss / 200:.4f}')
            running_loss = 0.0

            # 记录图像示例
            writer.add_images('input_images', inputs[:8], epoch * len(train_loader) + i)

    # 验证集
    model2.eval()
    running_val_loss = 0.0
    correct1 = correct2 = 0
    total1 = total2 = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model2(inputs)

            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total1 += labels.size(0)
            correct1 += (predicted == labels).sum().item()

    avg_val_loss = running_val_loss / len(val_loader)  # 这里应该是len(val_loader)而不是len(val_dataset)
    val_acc = correct1 / total1
    print(f'>>> Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_acc:.4f}')
    writer.add_scalar('val_loss', avg_val_loss, epoch)
    writer.add_scalar('val_accuracy', val_acc, epoch)

    # 早停机制：检查val loss是否负增长
    if avg_val_loss < best_val_loss:
        # 有改进，保存最佳模型
        best_val_loss = avg_val_loss
        best_epoch = epoch
        patience_counter = 0  # 重置计数器

        # 保存最佳模型
        torch.save({
            'model': model2.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'val_loss': avg_val_loss,
            'val_acc': val_acc
        }, 'res18_backbone_best.pth')
        print(f'>>> Best model saved with val_loss: {avg_val_loss:.4f}')
    else:
        # val loss负增长
        patience_counter += 1
        print(f'>>> Val loss not improved. Patience: {patience_counter}/{patience}')

        # 检查是否达到早停条件
        if patience_counter >= patience:
            print(f'>>> Early stopping triggered at epoch {epoch}!')
            print(f'>>> Best model was at epoch {best_epoch} with val_loss: {best_val_loss:.4f}')
            break  # 跳出训练循环


# 训练结束后保存最终模型（无论是否早停）
torch.save({
'model': model2.state_dict(),
'optimizer': optimizer.state_dict(),
'epoch': epoch
}, 'res18_backbone_final_3.pth')
writer.close()
print("Training complete")