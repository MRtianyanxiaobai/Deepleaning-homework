# *_*coding:utf-8 *_*
import numpy as  np
import matplotlib.pylab as plt
import sklearn
from dnnutill import *
from sklearn import datasets

def init_parameters(lay_dim):
    # 此时n_h为一个列表，表示着每一次的节点个数
    np.random.seed(3)
    parameters = {}
    for i in range(1, len(lay_dim)):
        # w[L]=n[L]*n[L-1] b[L] = n[L]
        parameters["w" + str(i)] = np.random.randn(lay_dim[i], lay_dim[i - 1])/np.sqrt(lay_dim[i - 1])
        parameters["b" + str(i)] = np.zeros((lay_dim[i], 1))

    return parameters;

# 测试
# lay_dim = [9, 7, 6, 5, 4, 3, 2, 1]
# paramers = init_parameters(lay_dim)
# for i in range(1, len(lay_dim)):
#     print(paramers["w" + str(i)].shape)

# 向前传播函数
# 主要分为两个步骤一个是线性计算wx+b，第二个步骤是激活函数
# 线性计算
def linear_forward(A_pre, W, b):
    Z = np.dot(W, A_pre) + b
    cache = (A_pre, W, b)
    return Z, cache

# 激活函数
def linear_activation_forward(A_pre, W, b, activation):
    if (activation == 'sigmod'):
        z, linear_cache = linear_forward(A_pre, W, b)
        A, activation_cache = sigmod(z)
    elif activation == 'relu':
        z, linear_cache = linear_forward(A_pre, W, b)
        A, activation_cache = relu(z)
    #A_pre,w,b,z
    cache=(linear_cache,activation_cache)

    return  A,cache

#多层模型的向前传播
def L_model_forward(X,parameters):
    caches=[]
    """
    cachae中有L-1个relu（从第0层到第L-2层）以及一个sigmod(第L-1层)
    
    """
    A =X
    #parameters既有w又有b，因此需要除以2
    L = len(parameters)//2
    for i in range(1,L):#1,2,3,4....L-1 =>放入cache的0-L-2
        A_pre = A
        A,cache = linear_activation_forward(A_pre, parameters["w"+str(i)], parameters["b"+str(i)],"relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters["w" + str(L)], parameters["b" + str(L)], "sigmod")
    caches.append(cache)

    return AL,caches


#test
# lay_dim = [9, 7, 6, 5, 4, 3, 2, 1]
# paramers = init_parameters(lay_dim)
# X = np.random.randn(9,100)
# Y =np.random.randint(0,2,100).reshape(1,-1)
# AL,caches = L_model_forward(X,paramers)



#成本函数的定义
def compute_cost(A2,Y):
    m = Y.shape[1]

    loss = np.multiply(Y,np.log(A2))+np.multiply(1-Y,np.log(1-A2))
    J = np.mean(loss)
    cost = float(np.squeeze(J))

    return -cost


#反向传播
def linear_backward(dz, cache):
    A_pre, w, b=cache
    m = A_pre.shape[1]
    dw = (1/m)*np.dot(dz,A_pre.T)
    db = (1/m)*np.sum(dz,axis=1,keepdims=True)
    dA_pre=np.dot(w.T,dz)

    return dA_pre,dw,db
def linear_activation_backward(dA,cache,activation="relu"):
    if (activation == 'sigmod'):
        linear_cache, activation_cache=cache
        dz = sigmoid_backward(dA,activation_cache)
        dA_pre, dw, db=linear_backward(dz, linear_cache)
    if (activation == 'relu'):
        linear_cache, activation_cache=cache
        dz = relu_backward(dA,activation_cache)
        dA_pre, dw, db=linear_backward(dz, linear_cache)

    return dA_pre,dw,db

#将整个反向传播连起来
def L_model_backforward(AL,Y,caches):
    #AL:表示通过正向传播第L层（最后一层）输出的A值
    L = len(caches)
    grads={}
    m = Y.shape[0]
    #divide是除法的意思
    dal = -np.divide(Y,AL)+np.divide(1-Y,1-AL)

    curren_cache = caches[L-1]
    grads["dA"+str(L)], grads["dw"+str(L)], grads["db"+str(L)]=linear_activation_backward(dal,curren_cache,activation="sigmod")
    #开始往后迭代 0,1,2,3,...L-2
    for i in reversed(range(L-1)):
        curren_cache = caches[i]
        da_p_t,dw_t,db_t = linear_activation_backward(grads["dA"+str(i+2)], curren_cache,activation="relu")
        grads["dA"+str(i+1)] = da_p_t
        grads["dw" + str(i + 1)] = dw_t
        grads["db" + str(i + 1)] = db_t

    return grads

def update_parameters(parameters,grads,learning_rate):
    L = len(parameters)//2
    for i in range(L):
        parameters["w"+str(i+1)]=parameters["w"+str(i+1)] - learning_rate*grads["dw"+str(i+1)]
        parameters["b" + str(i + 1)] = parameters["b" + str(i + 1)] - learning_rate * grads["db" + str(i + 1)]
    return parameters


# #测试
#test
# lay_dim = [9, 7, 6, 5, 4, 3, 2, 1]
# paramers = init_parameters(lay_dim)
# X = np.random.randn(9,100)
# Y =np.random.randint(0,2,100).reshape(1,-1)
# AL,caches = L_model_forward(X,paramers)
# grads = L_model_backforward(AL,Y,caches)



