# # pandas基本介绍
#
# import pandas as pd
# import numpy  as np
# s = pd.Series([1,3,6,np.nan,44,1])
# print(s)
#
# dates = pd.date_range('20160101',periods=6)
# print(dates)
#
# df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=['a','b','c','d'])
# print(df)
#
# df1 = pd.DataFrame(np.arange(12).reshape((3,4)))
# print(df1)
#
# #字典方式：df2 = pd.DataFrame({'A':1.,'B':pd.Timestamp('20130102'),'C':pd.Series(1,index=list(range(4)))})
#
# print(df1.dtypes)
# print(df1.index)
#
# print(df1.columns)
# print(df1.describe())
#
# print(df1.T)
#
# print(df1.sort_index(axis=1,ascending=False))


# #pandas 选择数据
# #进行数字的筛选
# #三种方式
# import pandas as pd
# import numpy as np
#
# dates = pd.date_range('20130101',periods=6)
# df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates,columns=['A','B','C','D'])
# print(df)
#
# #print(df.A)
# #print(df['A'])
# #print(df[0:3],df['20130102':'20130104'])
#
# #select by label:loc
# #print(df.loc['20130102'])
# #print(df.loc[:,['A','B']])
# #print(df.loc['20130102',['A','B']])
#
# #select by position:iloc
# #print(df.iloc[[1,3,5],1:3])
#
# # mixed selection:ix
# #print(df.ix[:3,['A','C']])
#
# #Boolean indexing
# #print(df)
# #print(df[df.A>8])

#pandas设置值
# import pandas as pd
# import numpy as np
#
# dates = pd.date_range('20130101',periods=6)
# df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates,columns=['A','B','C','D'])
#
# df.iloc[2,2] = 1111
# df.loc['20130101','B'] = 2222
# df.A[df.A>4] = 0
# df['F']=np.nan
# df['E'] = pd.Series([1,2,3,4,5,6],index = pd.date_range('20130101',periods=6))
# print(df)


#pandas 处理丢失数据

# import pandas as pd
# import numpy as np
#
# dates = pd.date_range('20130101',periods=6)
# df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates,columns=['A','B','C','D'])
# df.iloc[0,1] = np.nan
# df.iloc[1,2] = np.nan
#
# #print(df.dropna(axis=0,how='any'))#how = {'any','all'}
#
# #print(df.fillna(value=0))
# print(df.isnull())


#pandas 导入导出

# import pandas as pd
# data = pd.read_csv('student.csv')
# print(data)
#
# data.to_pickle('student.pickle')

#pandas 合并concat

import pandas as pd
import numpy as np

#concatenating

# df1 = pd.DataFrame(np.ones((3,4))*0,columns=['a','b','c','d'])
# df2 = pd.DataFrame(np.ones((3,4))*1,columns=['a','b','c','d'])
# df3 = pd.DataFrame(np.ones((3,4))*2,columns=['a','b','c','d'])
#
# print(df1)
# print(df2)
# print(df3)
#
# res = pd.concat([df1,df2,df3],axis=0,ignore_index=True)
# print(res)

#join,['inner','outer']
# df1 = pd.DataFrame(np.ones((3,4))*0,columns=['a','b','c','d'],index=[1,2,3])
# df2 = pd.DataFrame(np.ones((3,4))*1,columns=['b','c','d','e'],index=[2,3,4])
# print(df1)
# print(df2)
# #res = pd.concat([df1,df2],axis=1,join_axes=[df1.index])
# print(res)

# append

# df1 = pd.DataFrame(np.ones((3,4))*0,columns=['a','b','c','d'])
# df2 = pd.DataFrame(np.ones((3,4))*1,columns=['a','b','c','d'])
# #df2 = pd.DataFrame(np.ones((3,4))*1,columns=['a','b','c','d'],index=[2,3,4])
# #res = df1.append((df2,ignore_index = True))
# res = df1.append([df2,df1])
# print(res)


# pandas 合并 merge

#pandas合并merge

import pandas as  pd
#merging two df by key/keys.(may be used in database)
#simple example
# left = pd.DataFrame({'key':['K0','K1','K2','K3'],
#                      'A':['A0','A1','A2','A3'],
#                      'B':['B0','B1','B2','B3']})
# right = pd.DataFrame({'key':['K0','K1','K2','K3'],
#                      'C':['C0','C1','C2','C3'],
#                      'D':['D0','D1','D2','D3']})
# print(left)
# print(right)
#
# res = pd.merge(left,right,on='key')
# print(res)


#consider two keys
# left = pd.DataFrame({'key':['K0','K1','K2','K3'],
#                      'key2':['K0','K1','K0','K1'],
#                      'A':['A0','A1','A2','A3'],
#                      'B':['B0','B1','B2','B3']})
# right = pd.DataFrame({'key':['K0','K1','K2','K3'],
#                       'key2':['K0','K0','K0','K0'],
#                      'C':['C0','C1','C2','C3'],
#                      'D':['D0','D1','D2','D3']})
# print(left)
# print(right)
# # how = pd.merge('left','right','inner','outer']
# res = pd.merge(left,right,on=['key','key2'])
# print(res)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#plot data

#Series
# data = pd.Series(np.random.randn(1000),index=np.arange(1000))
# data = data.cumsum()

#DataFrame
data = pd.DataFrame(np.random.randn(1000,4),
                    index=np.arange(1000),
                    columns=list("ABCD"))
data = data.cumsum()
#plot methods:
#'bar','hist','box','kda','area','scatter','hexbin','pie'

ax = data.plot.scatter(x='A',y='B',color='DarkBlue')
data.plot.scatter(x='A',y='C',color='DarkGreen',ax=ax)

plt.show()



# data.plot()
# plt.show()
# print(data.head())
# # data.plot()

# ,lable = 'Class 1'
# ,lable = 'Class 2'