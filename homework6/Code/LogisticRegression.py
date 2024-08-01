import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 逻辑回归类
class LogisticRegression:
    def __init__(self, input_dim, lr=0.1):
        np.random.seed(0)
        self.weights = np.random.randn(input_dim, 1) * np.sqrt(2. / input_dim)
        self.bias = np.zeros((1, 1))
        self.lr = lr

    # Sigmoid激活函数
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # 前向传播过程
    def forward(self, x):
        z = np.dot(x, self.weights) + self.bias
        return self.sigmoid(z)

    # 计算损失
    def loss(self, y_pred, y):
        m = y.shape[0]
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    # 反向传播，更新参数
    def backprop(self, x, y):
        m = x.shape[0]
        y_pred = self.forward(x)
        dz = y_pred - y
        d_weights = np.dot(x.T, dz) / m
        d_bias = np.sum(dz) / m
        # 更新参数
        self.weights -= self.lr * d_weights
        self.bias -= self.lr * d_bias

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
model = LogisticRegression(input_dim)
epochs = 3000
batch_size = 32
losses = []

best_loss = float('inf')
best_epoch = 0
best_weights = None
best_bias = None

for epoch in range(epochs):
    for i in range(0, X.shape[0], batch_size):
        x_batch = X.iloc[i : i + batch_size].values
        y_batch = Y.iloc[i : i + batch_size].values.reshape(-1, 1)
        # 计算损失
        y_pred = model.forward(x_batch)
        current_loss = model.loss(y_pred, y_batch)
        # 反向传播
        model.backprop(x_batch, y_batch)
    
    losses.append(current_loss)
    
    if current_loss < best_loss:
        best_loss = current_loss
        best_epoch = epoch + 1
        best_weights = model.weights.copy()
        best_bias = model.bias.copy()
    
    print(f"Epoch: {epoch + 1}, Loss: {current_loss}")

# 使用最佳参数进行预测
model.weights = best_weights
model.bias = best_bias

results = (model.forward(X) > 0.5).astype(np.int32)
accuracy = (results == np.array(Y).reshape(-1, 1)).astype(np.int32).mean()
print(f"最终准确率：{accuracy * 100:.2f}%，最佳loss：{best_loss}")

# 可视化
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs + 1), losses, label="训练损失")
plt.axvline(best_epoch, color="green", linestyle="--", label=f"最佳Epoch: {best_epoch}")
plt.xlabel("迭代次数")
plt.ylabel("损失值")
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
ax2.scatter(feature1[results == 1], feature2[results == 1], color="blue", label="已购买")
ax2.set_title("预测结果")
ax2.set_xlabel("年龄")
ax2.set_ylabel("收入")
ax2.legend()

plt.show()
