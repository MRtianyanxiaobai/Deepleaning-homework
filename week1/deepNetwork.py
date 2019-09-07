# *_*coding:utf-8 *_*
#根据deeputil创建deepwork
import numpy as  np
import matplotlib.pylab as plt
from deepNetworkUtil import *
def L_layer_model(X,Y,layer_dims,learning_rate=0.0075,num_iterations=3000):
    np.random.seed(1)
    costs =[]
    xs=[]
    paramers = init_parameters(layer_dims)

    for i in range(num_iterations):
        AL,caches = L_model_forward(X,paramers)
        cost = compute_cost(AL,Y)
        grads = L_model_backforward(AL,Y,caches)
        paramers = update_parameters(paramers,grads,learning_rate)

        if i%100 == 0:
            xs.append(i)
            costs.append(cost)
            print("第",i,"次迭代。成本为："+str(cost))


    #迭代完成后绘制损失图
    plt.plot(xs,np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iteration')
    plt.show()
    return paramers

def predict(X,Y,paramers):
    AL, caches = L_model_forward(X, paramers)
    pred_y = np.array(AL>0.5,dtype=np.int)
    score = np.mean(pred_y==Y)
    print("准确率为:"+str(score))

    return pred_y


#测试
from lr_utils import load_dataset
#(209, 64, 64, 3)

#数据加载


# train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
# test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T


train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()


#现在trainx的维度为m*w*h*c，因此这里需要转换一下
train_set_x_orig = train_set_x_orig.transpose(1,2,3,0)
test_set_x_orig = test_set_x_orig.transpose(1,2,3,0)
train_set_x_orig = train_set_x_orig.reshape(12288,-1)
test_set_x_orig=test_set_x_orig.reshape(12288,-1)
#归一化处理

layers_dims = [12288, 20, 7, 5, 1]
train_set_x_orig = train_set_x_orig/255.0
test_set_x_orig = test_set_x_orig/255.0


paramers = L_layer_model(train_set_x_orig, train_set_y_orig, layers_dims,num_iterations=2500)

predict(train_set_x_orig,train_set_y_orig,paramers)
predict(test_set_x_orig,test_set_y_orig,paramers)