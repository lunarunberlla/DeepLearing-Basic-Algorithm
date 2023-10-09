import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class BasisFunction:
    def identity_basis(self, x):
        return x.unsqueeze(1)

    def multinomial_basis(self, x, feature_num=3):
        feat = [x.unsqueeze(1)]
        for i in range(2, feature_num+1):
            feat.append(x.unsqueeze(1)**i)
        return torch.cat(feat, dim=1)

    def gaussian_basis(self, x, feature_num=100):
        centers = np.linspace(0, 25, feature_num)
        width = 1.0 * (centers[1] - centers[0])
        x = x.unsqueeze(1)
        x = x.repeat(1, feature_num)
        out = (x - torch.Tensor(centers)) / width
        return torch.exp(-0.5 * out ** 2)

class DataLoader:
    def __init__(self, filename, basis_func=BasisFunction().multinomial_basis):
        self.filename = filename
        self.basis_func = basis_func

    def load_data(self):
        xys = []
        with open(self.filename, 'r') as f:
            for line in f:
                xys.append(list(map(float, line.strip().split())))
            xs, ys = zip(*xys)
            xs, ys = torch.Tensor(xs), torch.Tensor(ys)
            o_x, o_y = xs, ys
            phi0 = torch.ones_like(xs).unsqueeze(1)
            phi1 = self.basis_func(xs)
            xs = torch.cat([phi0, phi1], dim=1)
            return (xs, ys), (o_x, o_y)

class LinearModel(nn.Module):
    def __init__(self, ndim):
        super(LinearModel, self).__init__()
        self.w = nn.Parameter(torch.rand(ndim,1) * 0.2 - 0.1)

    def forward(self, x):
        return x @ self.w


train_loader = DataLoader('train.txt')
(xs, ys), (o_x, o_y) = train_loader.load_data()

ndim = xs.shape[1]

model = LinearModel(ndim=ndim)

optimizer = optim.Adam(model.parameters(), lr=0.1)
criterion = nn.MSELoss()

def train_one_step(model, xs, ys):
    optimizer.zero_grad()
    y_preds = model(xs)
    #loss = torch.mean(torch.sqrt(1e-12+(ys-y_preds.squeeze())**2))
    loss=criterion(ys,y_preds.squeeze())
    loss.backward()
    optimizer.step()
    return loss

def predict(model, xs):
    with torch.no_grad():
        y_preds = model(xs)
    return y_preds

def evaluate(ys, ys_pred):
    std = torch.sqrt(torch.mean((ys - ys_pred) ** 2))/len(ys)
    return std.item()

for i in range(5000):
    loss = train_one_step(model, xs, ys)
    if i % 500 == 0:
        print(f'loss at step {i} is {loss.item():.4f}')

y_preds = predict(model, xs)
std = evaluate(ys, y_preds)
print(f'Training set: standard deviation between predicted and actual values: {std:.1f}')

test_loader = DataLoader('test.txt')
(xs_test, ys_test), (o_x_test, o_y_test) = test_loader.load_data()

y_test_preds = predict(model, xs_test)
std = evaluate(ys_test, y_test_preds)
print(f'Test set: standard deviation between predicted and actual values: {std:.1f}')

plt.plot(o_x, o_y, 'ro', markersize=3)
plt.plot(o_x_test, o_y_test, 'bo', markersize=3)
plt.plot(o_x_test, y_test_preds, 'k')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend(['train', 'test', 'pred'])
plt.show()
