import torch
import torch.nn as nn
from torch import optim
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class DataLoader:
    '''定义一个加载数据的类'''
    def __init__(self):
        pass

    def __str__(self):
        return 'we will back a tensor like ([[x]]，[y])'

    def gaussian_kernel_matrix(self, X, sigma=5.0):
        X = X.float()
        XX = torch.matmul(X, X.t())
        X2 = XX.diag().unsqueeze(1)
        dist_matrix = X2 - 2 * XX + X2.t()
        K = torch.exp(-dist_matrix / (2 * sigma ** 2))
        return K

    def load_data(self, filepath='./data/train_kernel.txt', kernal=None):
        with open(filepath) as f:
            x,y=[],[]
            f.readline()
            for line in f:
                line = line.strip().split()
                x1 = float(line[0])
                x2 = float(line[1])
                t = int(line[2])
                x.append([x1, x2])
                y.append([t])
            x_tensor = torch.tensor(x, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)

            if kernal is not None:
                x_tensor = self.gaussian_kernel_matrix(x_tensor)

        return x_tensor, y_tensor


class Model(nn.Module):
    '''在逻辑回归中依然使用该模型，但是会在后面更改算法，使其变成逻辑回归'''
    def __init__(self, ndim):
        super(Model, self).__init__()
        self.w = nn.Parameter(torch.rand(ndim, 1) * 0.2 - 0.1)

    def forward(self, x):
        return x @ self.w



class Params:
    def __init__(self, lamuda=0.01, train_path='./data/train_kernel.txt', testpath='./data/test_kernel.txt', kernal=None):
        self.lamuda = lamuda
        self.train_path = train_path
        self.testpath = testpath
        self.kernal = kernal


class Criterion:
    '''这个类里面包含了SVM损失，平方损失和交叉熵损失函数的计算'''

    def __init__(self, params):
        self.params = params
        self.x, self.y = DataLoader().load_data(params.train_path, kernal=params.kernal)
        self.te_x, self.te_y = DataLoader().load_data(params.testpath, kernal=params.kernal)
        ndim = self.x.shape[1]
        self.model = Model(ndim=ndim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.lamuda = params.lamuda

    def SVMc(self, y, y_pred):
        '''svm损失'''
        A = torch.ones_like(y)
        B = torch.zeros_like(y)
        return torch.sum(torch.max(B, A - y * y_pred)) + self.lamuda * sum(p.pow(2.0).sum() for p in self.model.parameters())

    def LinerC(self, y, y_pred):
        '''squared error'''
        y=y.squeeze()
        y_pred=y_pred.squeeze()
        loss = torch.sum(torch.sqrt(1e-12 + (y - y_pred.squeeze()) ** 2)) + self.lamuda * sum(p.pow(2.0).sum() for p in self.model.parameters())
        return loss

    def Logisticc(self, y, y_pred):
        '''交叉熵损失'''
        y = y.float()
        y_pred = y_pred.float()
        y_pred = torch.sigmoid(y_pred)
        loss = torch.sum(torch.log(1+torch.exp(-y*y_pred)))+self.lamuda * sum(p.pow(2.0).sum() for p in self.model.parameters())
        return loss



class Utils(Criterion):

    '''这个类里面有一些我们评估模型的方法，以及训练方法，使用模型进行预测'''

    def __init__(self, params):
        super(Utils, self).__init__(params)

    def train_one_step(self, xs, ys, lossfunction,mode='SVM'):
        self.optimizer.zero_grad()
        if mode !='Logistic':
            y_preds = self.model(xs)
            loss = lossfunction(ys, y_preds)
            loss.backward()
            self.optimizer.step()
            return loss
        else:
            y_preds = torch.sigmoid(self.model(xs))
            loss = lossfunction(ys, y_preds)
            loss.backward()
            self.optimizer.step()
            return loss

    def predict(self, xs):
        with torch.no_grad():
            y_preds = self.model(xs)
        return y_preds

    def evaluate(self, ys, ys_pred,mode='SVM'):
        if mode!='Logistic':

            ys_pred = torch.where(ys_pred > 0, torch.ones_like(ys_pred), torch.zeros_like(ys_pred))
            ys = torch.where(ys > 0, torch.ones_like(ys), torch.zeros_like(ys))
            diff_count = (len(ys) - torch.sum(torch.logical_not(torch.eq(ys, ys_pred)))).float()
            C = torch.tensor([len(ys)], dtype=torch.float32)
        else:
            ys_pred = torch.sigmoid(ys_pred)
            ys_pred = torch.where(ys_pred >0.5, torch.ones_like(ys_pred), torch.zeros_like(ys_pred))
            ys = torch.where(ys > 0, torch.ones_like(ys), torch.zeros_like(ys))
            diff_count = (len(ys) - torch.sum(torch.logical_not(torch.eq(ys, ys_pred)))).float()
            C = torch.tensor([len(ys)], dtype=torch.float32)
        return diff_count/C


class Ways(Utils):
    def __init__(self, params):
        super(Ways, self).__init__(params)

    def Linear(self):
        for i in range(2500):
            loss = self.train_one_step(self.x, self.y, self.LinerC,mode='Linear')
            if i % 500 == 0:
                print(f'loss at step {i} is {loss.item():.4f}')
        y_preds = self.predict(self.x)
        std = self.evaluate(self.y, y_preds,mode='Linear')
        print(f'Linear->Training set: precision= {std}')
        y_test_preds = self.predict(self.te_x)
        std = self.evaluate(self.te_y, y_test_preds)
        print(f'Linear->Test set: precision={std}')

    def SVM(self):
        for i in range(2500):
            loss = self.train_one_step(self.x, self.y, self.SVMc,mode='SVM')
            if i % 500 == 0:
                print(f'loss at step {i} is {loss.item():.4f}')
        y_preds = self.predict(self.x)
        std = self.evaluate(self.y, y_preds,mode='SVM')
        print(f'SVM->Training set: precision= {std}')
        y_test_preds = self.predict(self.te_x)
        std = self.evaluate(self.te_y, y_test_preds)
        print(f'SVM->Test set: precision={std}')

    def Logistic(self):
        for i in range(2500):
            loss = self.train_one_step(self.x, self.y, self.Logisticc,mode='Logistic')
            if i % 500 == 0:
                print(f'loss at step {i} is {loss.item():.4f}')
        y_preds = self.predict(self.x)
        std = self.evaluate(self.y, y_preds,mode='Logistic')
        print(f'Logistic->Training set: precision= {std}')
        y_test_preds = self.predict(self.te_x)
        std = self.evaluate(self.te_y, y_test_preds,mode='Logistic')
        print(f'Logistic->Test set: precision={std}')



class RunProcess:
    def __init__(self, params):
        self.params = params
        self.crit = Criterion(params)
        self.comp = Ways(params)

    def run(self,name='SVM'):

        if name=='SVM':
            self.comp.SVM()
        elif name=='Linear':
            self.comp.Linear()
        elif name=='Logistic':
            self.comp.Logistic()
        else:
            print('prams error')


if __name__=='__main__':
    ''' --------------------题目一：使用高斯核函数解决线性不可分问题-----------------------------------------------'''
    params = Params(lamuda=0.01, train_path='./data/train_kernel.txt', testpath='./data/test_kernel.txt',
                   kernal='Gauss')
    process = RunProcess(params)
    process.run()
    # out##
    # SVM->Training set: precision= tensor([0.9750])
    # SVM->Test set: precision=tensor([0.9100])
    '''------------------------------------------------------------------------------------------------------'''

    '''---------------------题目二：分别使用线性分类器、logistic 回归以及SVM解决线性二分类问题，并比较三种模型的效果--------'''
    params = Params(lamuda=0.01, train_path='./data/train_linear.txt', testpath='./data/test_linear.txt',
                    kernal=None)
    process = RunProcess(params)
    process.run("SVM")
    process.run("Logistic")
    process.run("Linear")
    # out
    # SVM->Training set: precision= tensor([0.9550])
    # SVM->Test set: precision=tensor([0.9750])

    # Logistic->Training set: precision= tensor([0.9550])
    # Logistic->Test set: precision=tensor([0.9600])

    # Linear->Training set: precision= tensor([0.9500])
    # Linear->Test set: precision=tensor([0.9650])

    # Process finished with exit code 0


