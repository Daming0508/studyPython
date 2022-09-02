import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


n_data = torch.ones(100,2)                      #class0 x data (tensor),shape=(100,2)
x0 = torch.normal(2*n_data,1)                   #class0 y data (tensor),shape=(100,1)
y0 = torch.zeros(100)                           #class1 x data (tensor),shape=(100,1)
x1 = torch.normal(-2*n_data,1)                  #class1 y data (tensor),shape=(100,1)
y1 = torch.ones(100)
x = torch.cat((x0,x1),0).type(torch.FloatTensor)       #FloatTensor = 32-bit floating
y = torch.cat((y0,y1),).type(torch.LongTensor)         #LongTensor = 64-bit integer

#x,y = Variable(x),Variable(y)      #放入神经网络中进行学习


class Net(torch.nn.Module):  # 搭建神经网络
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x

net = Net(n_feature=2, n_hidden=10, n_output=2)   #第一位，两个特征，一个x轴特征一个y轴特征；10个神经元；设置输出也两个也两个特征 第三位
print(net)

optimizer = torch.optim.SGD(net.parameters(),lr=0.02)   #优化器，优化的就是net中的参数，学习效率lr
loss_func = torch.nn.CrossEntropyLoss()                          #回归问题中，用均方误差MSELoss ，分类问题中用CrossEntropyLoss
plt.ion()
for t in range(100):
    # softmax转换成概率才能真正称为prediction
    out = net(x)
    loss = loss_func(out,y)     #计算误差

    optimizer.zero_grad()      #清除上一批的误差
    loss.backward()            #误差反向传递，告诉个节点要有多少梯度
    optimizer.step()           #把梯度施加到各个节点上
    if t%2 == 0:
        plt.cla()
        prediction = torch.max(out,1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=pred_y,s=100,lw=0,cmap='Blues')
        accuracy = sum(pred_y == target_y) / 200
        plt.text(1.5,-4,'Accuracy=%.2f' % accuracy,fontdict={'size':20,'color':'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()