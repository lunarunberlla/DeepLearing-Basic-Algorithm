import torch
import torch.nn as nn
import os
import numpy as np
from torch import optim

# 数据加载函数
def load_data(filepath):
    data = np.loadtxt(filepath,skiprows=1)
    # 标签是最后一列
    x = torch.tensor(data[:, :-1], dtype=torch.float32)
    #增加一行全为一的向量，作为bais
    ones = torch.ones(len(x), 1)
    x=torch.cat([x,ones],dim=1)
    y = torch.tensor(data[:, -1], dtype=torch.long)
    # 转换 y 为 0, 1, 2
    y = (y + 1)
    return x, y

class SVM(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SVM, self).__init__()
        self.w = nn.Parameter(torch.rand(input_size,num_classes)*0.2+0.1)

    def forward(self, x):
        h = x@self.w
        return h
# 定义损失函数，这里使用的是hinge loss
def svm_loss(output, target):
    all_rows_idx = torch.arange(target.shape[0])
    correct_class_scores = output[all_rows_idx, target].unsqueeze(1)  # [N, 1]
    margin = torch.ones_like(output)
    loss = output - correct_class_scores + margin  # [N, C]
    loss[all_rows_idx, target] = 0  # zero out the correct class scores
    loss = loss.max(dim=1)[0]  # [N,]
    return loss.mean()

# 加载训练数据和测试数据
x_train, y_train = load_data('./data/train_multi.txt')
x_test, y_test = load_data('./data/test_multi.txt')

# 创建模型
model = SVM(x_train.shape[1], 3)  # 有三个类别

# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# 训练
for epoch in range(10000):
    # Forward pass
    outputs = model(x_train)
    loss = svm_loss(outputs, y_train)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 1000 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 100, loss.item()))

# 测试
with torch.no_grad():
    outputs = model(x_test)
    predicted = torch.argmax(outputs, dim=1)
    correct = (predicted == y_test).sum().item()
    print('Test Accuracy of the model on the test DATA: {} %'.format((correct / y_test.size(0)) * 100))
