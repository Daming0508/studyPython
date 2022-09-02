import torch
import torch.utils.data as Data

BATCH_SIZE = 8

x = torch.linspace(1,10,10)   #this is x data (torch tensor)   1-10 十个点   1,2,3,4……  返回一个张量 数据之间等距
y = torch.linspace(10,1,10)   #this is y data (torch tensor)   10-1 十个点   10,9,8……

torch_dataset = Data.TensorDataset(x,y)   #使用x进行训练，使用y算误差
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,      #要不要随机打断我们的数据进行抽样，True是不同的数据，False是相同的顺序
    #num_workers=2,     #batch_x和batch_y需要两个进程进行提取      bug:运行不了就直接注释掉
)

for epoch in range(3):
    for step,(batch_x,batch_y) in enumerate(loader):   #enumerate作用是每一次loader在提取时会施加一个索引
        #training....
        print('Epoch:',epoch,'|Step:',step,'|batch x:',
          batch_x.numpy(),'|batch y:',batch_y.numpy())


