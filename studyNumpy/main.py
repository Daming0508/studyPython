# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# import torch
# print(torch.__version__)

# #使用numpy生成数组,得到array的类型
# import numpy as np
# import random
# t1=np.array([1,2,3])
# print(t1)
# print(type(t1))
#
# t2 = np.array(range(10))
# print(t2)
# print(type(t2))
#
# t3 = np.arange(4,10,2)
# print(t3)
# print(type(t3))
#
# print(t3.dtype)
# print("*"*100)
#
#
# #numpy中的数据类型
# t4 = np.array(range(1,4),dtype=float)
# print(t4)
# print(t4.dtype)
#
# #numpy中的bool类型
#
# t5 = np.array([1,1,0,1,0,0],dtype=bool)
# print(t5)
# print(t5.dtype)
#
# #调整数据类型
# t6 = t5.astype("i1")
# print(t6)
# print(t6.dtype)
#
# #numpy中的小数
# t7 = np.array([random.random() for i in range(10)])
# print(t7)
# print(t7.dtype)
#
#
# t8 = np.round(t7,2)
# print(t8)

# import pandas as pd
# print((pd.__version__))
import numpy as np

# array = np.array([[1,2,3],[2,3,4]])
# print(array)
#
# print('number of dim:',array.ndim)
# print('shape:',array.shape)
# print('size:',array.size)

# a = np.array([2,3,4],dtype=np.int64)
# print(a.dtype)
#
# b = np.array([[2,3,4],
#               [3,4,5]])
# print(b)
#
# c = np.zeros((3,4))
# print(c)
#
# d = np.ones((3,4),dtype=np.int64)
# print(d)
#
# e = np.empty((3,4))
# print(e)
#
# f = np.arange(10,20,2)
# print(f)
#
# g = np.arange(12).reshape((3,4))
# print(g)
#
# h = np.linspace(1,10,20)
# print(h)
#
# i = np.linspace(1,10,6).reshape((2,3))
# print(i)
#
# j = np.array([10,20,30,40])
# k = np.arange(4)
# print(j,k)
# l = j - k
# m = b**2
# print(l)
# print(m)
#
# #n = 10*np.sin(j)
# #print(n)
# print(k<3)
#
# o = np.array([[1,1],
#               [0,1]])
# p = np.arange(4).reshape(2,2)
# print(o)
# print(p)
#
# #逐个相乘
# q = o*p
# #矩阵法则相乘
# q_dot = np.dot(o,p)
# q_dot_2 = o.dot(p)
# print(q)
# print(q_dot_2)
#
# a = np.random.random((2,4))
#
# # print(a)
# # print(np.sum(a,axis=1))
# # print(np.min(a,axis=0))
# # print(np.max(a,axis=1))
#
# #numpy的基础运算
# A = np.arange(2,14).reshape((3,4))
# print(A)
# print(np.argmin(A))
# print(np.argmax(A))
# print(np.average(A))
# print(np.median(A))
# print(np.cumsum(A))
# print(np.diff(A))
# print(np.nonzero(A))
#
# B = np.arange(14,2,-1).reshape((3,4))
# print(B)
# print(np.transpose(B))
# print(B.T)
# print((B.T).dot(B))
# print(np.clip(B,5,9))
# #print(np.mean(B,axis=1)   计算平均值

#numpy的索引
# A = np.arange(3,15).reshape((3,4))
# print(A)
# print(A[2])
# print(A[1][1])
# print(A[2,1])
# print(A[2:])
# print(A[:1])
#
# for row in A:
#     print(row)
#
# for column in A.T:
#     print(column)
#
# for item in A.flat:
#     print(item)


#numpy的array合并

# A = np.array([1,1,1])
# B = np.array([2,2,2])
# C = np.vstack((A,B))  #vertical stack
# D = np.hstack((A,B))  #HORIZONTAL STACK
# print(A.shape,D.shape)
# print(A[:,np.newaxis])
#
# E = np.concatenate((A,B,B,A),axis=0)
# print(E)


# #numpy的array分割
#
# A = np.arange(12).reshape((3,4))
# print(A)
# print(np.split(A,2,axis=1))
# print(np.split(A,3,axis=0))
# print(np.array_split(A,3,axis=1))
# print(np.vsplit(A,3))
# print(np.hsplit(A,2))

# #numpy的copy&deep copy
#
# a = np.arange(4)
# print(a)
# b = a
# c = a
# d = b
# a[0] = 11
# print(a)
# print(b)
# print( d is b)
#
# d[1:3]=[22,33]
# print(a,b,c,d)
#
# b = a.copy() #deep copy
# print(b)
# a[3] = 44
# print(a)
# print(b)