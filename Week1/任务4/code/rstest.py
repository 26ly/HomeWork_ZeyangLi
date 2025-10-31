import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import pandas as pd
from tqdm import tqdm

# -------------------------
# 配置
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "res18_backbone_final_3.pth"
test_folder = "C:\\Users\\17765\\mmpretrain\\projects\\RoboM\\DataSet\\test"
num_classes = 8

# -------------------------
# 预处理
# -------------------------
pre_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# -------------------------
# 加载模型
# -------------------------
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


# -------------------------
# 加载模型和测试集
# -------------------------
model = MyModel().to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()

# 加载测试集
test_dataset = ImageFolder(root=test_folder, transform=pre_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

print(f"测试集大小: {len(test_dataset)}")
print(f"测试集类别: {test_dataset.classes}")


# -------------------------
# 测试函数
# -------------------------
def evaluate_model(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="测试中"):
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 统计
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            probabilities = torch.softmax(outputs, dim=1)

            # 保存预测结果
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # 计算指标
    avg_loss = running_loss / total
    accuracy = correct / total

    return avg_loss, accuracy, all_predictions, all_labels, all_probabilities


# -------------------------
# 执行测试
# -------------------------
criterion = nn.CrossEntropyLoss()

print("开始测试...")
test_loss, test_accuracy, predictions, true_labels, probabilities = evaluate_model(
    model, test_loader, criterion
)

# -------------------------
# 输出结果
# -------------------------
print("\n" + "=" * 50)
print("测试结果:\n")
print(f"Loss: {test_loss:.4f}")
print(f"Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
print(f"正确样本数: {int(test_accuracy * len(test_dataset))} / {len(test_dataset)}")

# -------------------------
# 各类别详细准确率
# -------------------------
from sklearn.metrics import classification_report, confusion_matrix

print("\n各类别准确率:")
class_names = test_dataset.classes
print(classification_report(true_labels, predictions,
                            target_names=class_names, digits=4))

# -------------------------
# 混淆矩阵
# -------------------------
cm = confusion_matrix(true_labels, predictions)
print("\n混淆矩阵:")
print(cm)