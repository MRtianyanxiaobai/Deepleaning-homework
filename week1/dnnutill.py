# *_*coding:utf-8 *_*
import numpy as  np
#激活函数有两种Relu函数和sigmod函数，下面想定义这两个函数以及他们的导数
def sigmod(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache

def sigmoid_backward(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ

def relu(Z):
    A = np.maximum(0,Z)
    cache = Z
    return A,cache

def relu_backward(dA,cache):
    Z = cache
    #dz = dA * g'
    #g'=0 or 1
    dz = np.array(dA,copy=True)

    dz[Z<=0]=0
    return dz