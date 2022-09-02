# method 1
import torch.nn


class Net(torch.nn.Module):  # 搭建神经网络
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)       #给了一个class属性叫hidden
        self.out = torch.nn.Linear(n_hidden, n_output)           #给了一个class属性叫out

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x


net1 = Net(n_feature=2, n_hidden=10, n_output=2)
print(net1)


#method 2
net2 = torch.nn.Sequential(
    torch.nn.Linear(2,10),                   #输入层
    torch.nn.ReLU(),                         #激活函数，大写RelU作为一个类，输出的时候有名字
    torch.nn.Linear(10,2),                   #输出层
)
print(net2)