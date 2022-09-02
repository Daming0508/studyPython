import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import  matplotlib.pyplot as plt

#Hyper Parameters
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),   #压缩图片至(0,1)   图片（0-255）
    download=DOWNLOAD_MNIST
)
#plot one example
# print(train_data.train_data.size())  #(60000,28,28)
# print(train_data.train_labels.size()) #(60000)
# plt.imshow(train_data.train_data[0].numpy(),cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()

train_loader = Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)

test_data = torchvision.datasets.MNIST(root='./mnist',train=False) #false意味着提取出来的是测试数据
test_x = Variable(torch.unsqueeze(test_data.test_data,dim=1),volatile=True).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.test_labels[:2000]           #取前两千个

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(                            #(1, 28, 28)  卷积层
                in_channels=1,                    #这张图片是有多少个层的高度
                out_channels=16,                  #过滤器的高度（一共多少个过滤器）
                kernel_size=5,                    #说明过滤器宽和高都是五个像素点
                stride=1,                         #每过一次的高度
                padding=2,                        #如果过滤器超过了图片，在图片周围加上零.padding = (kernel_size-1)/2 = (5-1)/2
            ),   #过滤器      →(16, 28, 28)
            nn.ReLU(),       #→(16, 28, 28)
            nn.MaxPool2d(kernel_size=2),         #选更重要的信息   →(16, 14, 14)  池化层
        )
        self.conv2 = nn.Sequential(              #→(16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),          #→(32, 14, 14)
            nn.ReLU(),                           #→(32, 14, 14)
            nn.MaxPool2d(2)                      #→(32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7,10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)                       #(batch, 32, 7, 7)   考虑了batch
        x = x.view(x.size(0),-1)                #(batch, 32*7*7)
        output = self.out(x)
        return output

cnn = CNN()
print(cnn)   #net architecture

optimizer = torch.optim.Adam(cnn.parameters(),lr=LR)    #optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                    #the target lablel is not one-hotted

#training and testing
for epoch in range(EPOCH):
    for step,(x,y) in enumerate(train_loader):
        b_x = Variable(x)
        b_y = Variable(y)

        output = cnn(b_x)
        loss = loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            test_out = cnn(test_x)
            pred_y = torch.max(test_out,1)[1].data.squeeze()
            accuracy = sum(pred_y == test_y) / test_y.size(0)     #计算有百分之多少图片预测对了
            print('Epoch:',epoch,'|train loss:%.4f'% loss.data.numpy(),'| test accuracy: %.2f' % accuracy)

#print 10 predictions from test data
test_out = cnn(test_x[:10])
pred_y = torch.max(test_out,1)[1].data.numpy().squeeze()
print(pred_y,'prediction number')
print(test_y[:10].numpy(),'real number')
