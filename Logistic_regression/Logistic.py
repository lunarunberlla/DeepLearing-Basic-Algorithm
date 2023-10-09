import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class DataLoader:
    def __init__(self):
        pass

    def __str__(self):
        return "we will back two tensor like [[x1,x2],][y]]"
    def create_dataset(self,dot_num=100):
        x_p = torch.normal(3., 1, size=(dot_num,))
        y_p = torch.normal(6., 1, size=(dot_num,))
        y = torch.ones(dot_num)
        C1 = torch.stack([x_p, y_p, y], dim=1)

        x_n = torch.normal(6., 1, size=(dot_num,))
        y_n = torch.normal(3., 1, size=(dot_num,))
        y = torch.zeros(dot_num)
        C2 = torch.stack([x_n, y_n, y], dim=1)

        plt.scatter(C1[:, 0], C1[:, 1], c='b', marker='+')
        plt.scatter(C2[:, 0], C2[:, 1], c='g', marker='o')
        data_set = torch.cat((C1, C2), dim=0)
        data_set = data_set[torch.randperm(data_set.size(0))]  # shuffle the data_set
        x,y=[],[]
        for line in data_set:
            x1=float(line[0])
            x2=float(line[1])
            t=float(line[2])
            x.append([x1,x2])
            y.append([t])
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        y_tensor= torch.where(y_tensor>0, torch.ones_like(y_tensor), torch.zeros_like(y_tensor)-1)
        ones = torch.ones(len(x_tensor), 1)
        x_tensor = torch.cat([x_tensor, ones], dim=1)
        return x_tensor,y_tensor


class Model(nn.Module):
    '''在逻辑回归中依然使用该模型，但是会在后面更改算法，使其变成逻辑回归'''
    def __init__(self, ndim):
        super(Model, self).__init__()
        self.w = nn.Parameter(torch.rand(ndim, 1) * 0.2 - 0.1)

    def forward(self, x):
        return torch.sigmoid(x @ self.w)

class Params:
    def __init__(self, lamuda=0.01,):
        self.lamuda = lamuda

class Criterion:
    '''这个类里面包含了SVM损失，平方损失和交叉熵损失函数的计算'''

    def __init__(self, params):
        self.params = params
        self.x, self.y = DataLoader().create_dataset()
        self.te_x, self.te_y = DataLoader().create_dataset()
        ndim = self.x.shape[1]
        self.model = Model(ndim=ndim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.lamuda = params.lamuda


    def Logisticc(self, y, y_pred):
        '''交叉熵损失'''
        y = y.float()
        y_pred = y_pred.float()
        y_pred = torch.sigmoid(y_pred)
        loss = torch.mean(torch.log(1+torch.exp(-y*y_pred)))+self.lamuda * sum(p.pow(2.0).sum() for p in self.model.parameters())
        return loss


class Utils(Criterion):

    '''这个类里面有一些我们评估模型的方法，以及训练方法，使用模型进行预测'''

    def __init__(self, params):
        super(Utils, self).__init__(params)

    def train_one_step(self, xs, ys, lossfunction):
        self.optimizer.zero_grad()
        y_preds = self.model(xs)
        loss = lossfunction(ys, y_preds)
        loss.backward()
        self.optimizer.step()
        return loss

    def predict(self, xs):
        with torch.no_grad():
            y_preds = self.model(xs)
        return y_preds

    def evaluate(self, ys, ys_pred):
        ys_pred = torch.where(ys_pred >0.5, torch.ones_like(ys_pred), torch.zeros_like(ys_pred))
        ys = torch.where(ys > 0, torch.ones_like(ys), torch.zeros_like(ys))
        diff_count = (len(ys) - torch.sum(torch.logical_not(torch.eq(ys, ys_pred)))).float()
        C = torch.tensor([len(ys)], dtype=torch.float32)
        return diff_count/C

class Ways(Utils):
    def __init__(self, params):
        super(Ways, self).__init__(params)


    def Logistic(self):
        for i in range(2500):
            loss = self.train_one_step(self.x, self.y, self.Logisticc)
            if i % 500 == 0:
                print(f'loss at step {i} is {loss.item():.4f}')
        y_preds = self.predict(self.x)
        std = self.evaluate(self.y, y_preds)
        print(f'Logistic->Training set: precision= {std}')
        y_test_preds = self.predict(self.te_x)
        std = self.evaluate(self.te_y, y_test_preds)
        print(f'Logistic->Test set: precision={std}')

class RunProcess:
    def __init__(self, params):
        self.params = params
        self.crit = Criterion(params)
        self.comp = Ways(params)

    def run(self):
        self.comp.Logistic()

if __name__=='__main__':
    ''' --------------------实现Logistic回归-----------------------------------------------'''
    params = Params(lamuda=0.01)
    process = RunProcess(params)
    process.run()
    # out
    # Logistic->Training set: precision= tensor([0.9800])
    # Logistic->Test set: precision=tensor([0.9900])