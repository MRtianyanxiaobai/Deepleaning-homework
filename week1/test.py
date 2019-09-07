# *_*coding:utf-8 *_*
import numpy as  np
import  matplotlib.pylab as plt
def sigmod(x):
    return 1/(1+np.exp(-x));

#阶越函数
def step_function(x):
    return np.array(x>0,dtype=np.int)

#ReLu函数
def relu(x):
    return np.maximum(0,x)

#softmax
def softmax(x):
    exp_a = np.exp(x)
    sum_ea = np.sum(exp_a)
    return x/sum_ea


# 恒等函数
def identity_function(x):
    return x

#初始化参数
def init_network():
    network={};
    network['w1'] = np.array([
            [0.1,0.1,0.5],
            [0.2,0.4,0.5],

    ])
    network['b1']=np.array([0.1,0.2,0.3])
    network['w2']= np.array([
            [0.1,0.2],
            [0.2,0.4],
            [0.3,0.1]
    ])
    network['b2'] = np.array([0.1, 0.3])

    network['w3']= np.array([
            [0.1,0.2],
            [0.3,0.4],

    ])
    network['b3'] = np.array([0.2, 0.1])
    return  network


def foeward(network,x):
    w1,w2,w3=network['w1'],network['w2'],network['w3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']





