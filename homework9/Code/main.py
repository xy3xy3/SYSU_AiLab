import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image


# 自定义Dataset类用于加载测试集
class TestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = []
        self.labels = []
        self.classes = ['baihe', 'dangshen', 'gouqi', 'huaihua', 'jinyinhua']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        for filename in os.listdir(root):
            for cls in self.classes:
                if filename.startswith(cls):
                    self.images.append(os.path.join(root, filename))
                    self.labels.append(self.class_to_idx[cls])
                    break

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")  # 使用 PIL 读取图像
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label
model_path = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/best_model.pth"
# 设置文件夹路径
train_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./train").replace("\\", "/")
test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./test").replace("\\", "/")

# 定义数据预处理
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载训练集
train_dataset = ImageFolder(root=train_dir, transform=transform)
test_dataset = TestDataset(root=test_dir, transform=transform)

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 更新模型结构
def main():
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    num_ftrs = model.fc.in_features
    # 添加Dropout层
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 5)
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # 使用Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 更新学习率调整策略
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    num_epochs = 50
    best_acc = 0.0
    train_acc_history = []
    test_acc_history = []
    train_loss_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total

        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}')

        # 更新学习率
        scheduler.step(epoch_loss)  # 使用验证损失来调整学习率

        # 测试模型
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_acc = correct / total
        test_acc_history.append(test_acc)
        print(f'Test Accuracy: {test_acc:.4f}')

        # 保存性能最好的模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), model_path)

    # 调用绘图函数
    plot_accuracy_curve(train_loss_history, train_acc_history, test_acc_history)


def load_and_test_model():
    global model_path, test_loader, device
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = correct / total
    print(f'Loaded model Test Accuracy: {test_acc:.4f}')
    return test_acc

def plot_accuracy_curve(train_loss_history, train_acc_history, test_acc_history):
    plt.figure()
    plt.plot(train_loss_history, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss Curve')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(train_acc_history, label='Train Accuracy')
    plt.plot(test_acc_history, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
    # load_and_test_model()
