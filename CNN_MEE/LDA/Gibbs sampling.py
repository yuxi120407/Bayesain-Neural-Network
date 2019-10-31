# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 16:10:54 2019

@author: Xi Yu
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import random
K = 2 #最终观测结果来自的硬币
V = 2 #观测结果的取值个数（0,1）

X = [1,1,1,1,1,1,0,0,0,0] #观测结果
N = len(X) #观测次数
Z= [0] * N #每次观测结果对应的硬币编号

nz=[0,0] #cnz[i]硬币A掷出i面的次数（使用硬币i的次数）
nxz=[[0,0],[0,0]] #nzx[i][j] 观测结果为i来自j硬币的次数
nxsum = N #观测总次数
p=[0,0] #gibbs采样条件概率分布
#%%

alpha = 1 #硬币A Beta分布的 alpha、beta超参数,这里直接取<1,1>
beta=1  #硬币i Beta分布的 alpha、beta超参数,这里直接取<1,1>

max_iter = 100 #迭代次数

def init_params():
    # initilize the latent parameters Z
    for i in range(N) :
        prob = random.random()
        if prob > 0.5:
            Z[i] = 1
        else:
            Z[i] = 0
    # 统计
    for x,z in zip(X,Z):
        nxz[x][z] += 1
        nz[z] += 1
#%%
def sample():
    for cur_iter in range(max_iter):
        for i,x in enumerate(X,0):
            #去除观测结果i之后的计数
            z = Z[i]
            nz[z] -= 1
            nxz[x][z] -= 1
            global nxsum
            nxsum -= 1
            
            #计算条件分布
            for k in range(K):
                p[k] = (nxz[x][k] + beta)/(nz[k] +  V * beta)*(nz[k] + alpha)/(nxsum + K * alpha)
            
            #采样
            for k in range(1,K):
                p[k] = p[k-1] + p[k] 
            prob = random.random() * p[-1] #这里要进行归一化
            for k in range(K):
                if prob < p[k]:
                    z = k 
                    break
			
            nz[z] += 1
            nxz[x][z] += 1
            nxsum += 1
            Z[i] = z
init_params()
sample()

print((nz[1] + alpha)/(nxsum + K * alpha)) #硬币A正面朝上的概率
print((nxz[1][1] + beta)/(nz[1] +  V * beta)) #硬币B正面朝上的概率
print((nxz[0][1] + beta)/(nz[0] +  V * beta)) #硬币C正面朝上的概率
