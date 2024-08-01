import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 模型类
class MLP:
    def __init__(self, input_dim, lr=0.1):
        np.random.seed(0)
        # [2,64] -> [64,32] -> [32,1]
        self.weights1 = np.random.randn(input_dim, 64) * np.sqrt(2. / input_dim)
        self.bias1    = np.zeros((1, 64))
        self.weights2 = np.random.randn(64, 32) * np.sqrt(2. / 64)
        self.bias2    = np.zeros((1, 32))
        self.weights3 = np.random.randn(32, 1) * np.sqrt(2. / 32)
        self.bias3    = np.zeros((1, 1))
        self.lr = lr

    # ReLu激活函数
    def relu(self, z):
        return np.maximum(0, z)

    # Sigmoid激活函数
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # 前向传播过程
    def forward(self, x):
        self.z1 = np.dot(x, self.weights1) + self.bias1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.relu(self.z2)
        self.z3 = np.dot(self.a2, self.weights3) + self.bias3
        self.a3 = self.sigmoid(self.z3)
        return self.a3

    # 计算损失
    def loss(self, y_pred, y):
        return np.mean((y_pred - y) ** 2)

    # 反向传播，更新参数
    def backprop(self, x, y):
        m = x.shape[0]
        # 计算梯度
        d_a3 = (self.a3 - y)
        d_z3 = d_a3 * (self.a3 * (1 - self.a3))
        d_weights3 = np.dot(self.a2.T, d_z3) / m
        d_bias3 = np.sum(d_z3, axis=0, keepdims=True) / m
        d_a2 = np.dot(d_z3, self.weights3.T)
        d_z2 = d_a2 * (self.z2 > 0)
        d_weights2 = np.dot(self.a1.T, d_z2) / m
        d_bias2 = np.sum(d_z2, axis=0, keepdims=True) / m
        d_a1 = np.dot(d_z2, self.weights2.T)
        d_z1 = d_a1 * (self.z1 > 0)
        d_weights1 = np.dot(x.T, d_z1) / m
        d_bias1 = np.sum(d_z1, axis=0, keepdims=True) / m
        # 更新参数
        self.weights1 -= self.lr * d_weights1
        self.bias1    -= self.lr * d_bias1
        self.weights2 -= self.lr * d_weights2
        self.bias2    -= self.lr * d_bias2
        self.weights3 -= self.lr * d_weights3
        self.bias3    -= self.lr * d_bias3

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号

# 数据预处理
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "data.csv")
data = pd.read_csv(data_path)
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]
X_min = X.min()
X_max = X.max()
X = (X - X_min) / (X_max - X_min)

# 模型训练
input_dim = X.shape[1]
model = MLP(input_dim)
epochs = 3000
batch_size = 32
losses = []
best_loss = float('inf')
best_epoch = 0
best_weights1 = None
best_weights2 = None
best_weights3 = None
best_bias1 = None
best_bias2 = None
best_bias3 = None

for epoch in range(epochs):
    for i in range(0, X.shape[0], batch_size):
        x_batch = X.iloc[i : i + batch_size].values  # [batchsize, 2]
        y_batch = Y.iloc[i : i + batch_size].values.reshape(-1, 1)  # [batchsize, 1]

        # 前向传播
        y_pred = model.forward(x_batch)  # [batchsize, 1]

        # 计算损失
        current_loss = model.loss(y_pred, y_batch)

        # 反向传播，更新参数
        model.backprop(x_batch, y_batch)

    losses.append(current_loss)
    if current_loss < best_loss:
        best_loss = current_loss
        best_epoch = epoch + 1
        best_weights1 = model.weights1.copy()
        best_weights2 = model.weights2.copy()
        best_weights3 = model.weights3.copy()
        best_bias1 = model.bias1.copy()
        best_bias2 = model.bias2.copy()
        best_bias3 = model.bias3.copy()

    print(f"Epoch: {epoch + 1}, Loss: {current_loss}")

# 使用最佳参数进行预测
model.weights1 = best_weights1
model.weights2 = best_weights2
model.weights3 = best_weights3
model.bias1 = best_bias1
model.bias2 = best_bias2
model.bias3 = best_bias3

results = (model.forward(X) > 0.5).astype(np.int32)
accuracy = (results == np.array(Y).reshape(-1, 1)).astype(np.int32).mean()
print(f"最终准确率：{accuracy * 100:.2f}%，最佳loss：{best_loss}")

# 可视化
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs + 1), losses, label="训练损失")
plt.axvline(best_epoch, color="green", linestyle="--", label=f"最佳Epoch: {best_epoch}")
plt.xlabel("迭代次数")
plt.ylabel("损失值")
plt.text(epochs, losses[-1], f"最终损失: {losses[-1]:.4f}", ha="right", va="baseline")
plt.legend()

results = results.flatten()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

feature1 = X.iloc[:, 0]
feature2 = X.iloc[:, 1]

ax1.scatter(feature1[Y == 0], feature2[Y == 0], color="red", label="未购买")
ax1.scatter(feature1[Y == 1], feature2[Y == 1], color="blue", label="已购买")
ax1.set_title("真实情况")
ax1.set_xlabel("年龄")
ax1.set_ylabel("收入")
ax1.legend()

ax2.scatter(feature1[results == 0], feature2[results == 0], color="red", label="未购买")
ax2.scatter(
    feature1[results == 1], feature2[results == 1], color="blue", label="已购买"
)
ax2.set_title("预测结果")
ax2.set_xlabel("年龄")
ax2.set_ylabel("收入")
ax2.legend()

plt.show()
