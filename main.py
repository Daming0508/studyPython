#numpy 还是 Torch

import torch
import numpy as np
import torch.nn.functional as F
from  torch.autograd import Variable
import matplotlib.pyplot as plt

# np_data = np.arange(6).reshape((2,3))
# torch_data = torch.from_numpy(np_data)
# tensor2array = torch_data.numpy()
# print(
#     '\nnumpy',np_data,
#     '\ntorch',torch_data,
#     '\ntensor2arry',tensor2array
# )

# #abs
# # data = [-1,-2,1,2]
# # tensor = torch.FloatTensor(data) #32bit
# # # http://pytorch.org/docs/torch.html#math-operations
# # print(
# #     '\nabs,'
# #     '\nnumpy',np.sin(data),
# #     '\ntorch',torch.sin(tensor)
# # )
#
# data = [[1,2],[3,4]]
# tensor = torch.FloatTensor(data)
# print(
#     '\nnumpy',np.matmul(data,data),#data.dot(data)
#     '\ntorch',torch.mm(tensor,tensor)
# )


#Variable 变量
# from torch.autograd import Variable
#
# tensor = torch.FloatTensor([[1,2],[3,4]])
# variable = Variable(tensor,requires_grad=True)
#
#
# t_out = torch.mean(tensor*tensor)
# v_out = torch.mean(variable*variable)
#
# # print(t_out)
# # print(v_out)
#
# v_out.backward()
# #v_out = 1/4*sum(var*var)
# #d(v_out)/d(var) = 1/4*2*variable = variable/2
# print(variable.grad)
# print(variable)
# print(variable.data)


#创建激励函数********************************************************************
# fake data
# x = torch.linspace(-5,5,200)
# x = Variable(x)
# x_np = x.data.numpy()
#
# y_relu = F.relu(x).data.numpy()
# y_sigmod = torch.sigmoid(x).data.numpy()
# y_tanh = torch.tanh(x).data.numpy()
# y_softplus = F.softplus(x).data.numpy()
#
# plt.figure(1,figsize=(8,6))
# plt.subplot(221)
# plt.plot(x_np,y_relu,c='red',label='relu')
# plt.ylim(((-1,5)))
# plt.legend(loc='best')
#
#
# plt.subplot(222)
# plt.plot(x_np,y_sigmod,c='red',label='sigmod')
# plt.ylim(((-0.2,0.2)))
# plt.legend(loc='best')
#
#
# plt.subplot(223)
# plt.plot(x_np,y_sigmod,c='red',label='tanh')
# plt.ylim(((-1.2,1.2)))
# plt.legend(loc='best')
#
# plt.subplot(224)
# plt.plot(x_np,y_sigmod,c='red',label='softplus')
# plt.ylim(((-0.2,6)))
# plt.legend(loc='best')
# plt.show()


#关系拟合(回归）

x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1) #x data (tensor),shape = (100,1)
y = x.pow(2) + 0.2*torch.rand(x.size())

x,y = Variable(x),Variable(y)

# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()

class Net(torch.nn.Module):
    def __init__(self,n_features,n_hidden,n_output):    # 搭建一个神经网络所需要的信息
        super(Net,self).__init__()  #这步以上都是官方流程 下面开始写自己需要的东西
        self.hidden = torch.nn.Linear(n_features,n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)  #预测的神经层


    def forward(self,x):    #前向传播的一个过程，将神经网络的信息放在这里进行组合，搭流程图
         x = F.relu(self.hidden(x))
         x = self.predict(x)
         return x
net = Net(n_features=1,n_hidden=10,n_output=1)      #define the network
print(net)


#优化神经网络
optimizer = torch.optim.SGD(net.parameters(),lr=0.2)
loss_func = torch.nn.MSELoss()

#为了可视化插入（1）*************
plt.ion()  #实时打印过程


#训练神经网络
for t in range(200):       #100步
    prediction = net(x)
    loss = loss_func(prediction,y)


    optimizer.zero_grad()
    loss.backward()    #反向传播计算出每个节点的梯度
    optimizer.step()   #开启优化器


    # plot and show learning process
    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)


plt.ioff()
plt.show()

#*************************************************************************************************************
