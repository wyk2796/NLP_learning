# coding:utf-8

import numpy as np
from nlp_modle import util

data_path='E:\\temp\data\\iris.data'
data_org = open(data_path, encoding='utf-8', mode='r').readlines()
data = []
classmap = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
for line in data_org:
    arr = line.strip().split(',')
    data.append([float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]), classmap[arr[4]]])


W = np.zeros(shape=(3, 4))
b = np.ones(shape=(3, ))
#W = np.random.normal(size=(3, 4))
#Wd = np.zeros(shape=(3, 5))

def forward(x, W):
    z2 = W.dot(x) + b
    # print(z2,'z2')
    a2 = signmod(z2)
    # print(a2, 'a2')
    Wd = derivative_signmod(a2)
    # print(Wd, 'Wd')
    z3 = softmax(a2)
    return z3, Wd


def signmod(z):
    return 1 / (1 + np.exp(-z))


def derivative_signmod(a):
    # print(1-a)
    # print(a * (1-a))
    return a * (1 - a)


def grad_computer(y_f, y, x, Wd):
    right_index = y.argmax()
    back = y_f.copy()
    back[right_index] -= 1
    return np.dot(np.reshape(x, newshape=[4, 1]), np.reshape(back * Wd, newshape=[1, 3])).T, back * Wd


def loss_cross_entropy(test_data):
    loss = 0
    count = 0
    for v in test_data:
        x = v[0:4]
        y = util.one_hot_simple(v[-1], 3)
        y_,_ = forward(x, W)
        if y_.argmax() == v[-1]:
            count += 1
        loss += -np.sum(y * np.log(y_))
    return loss / len(test_data), count / len(test_data)


def softmax(z):
    ze = np.exp(z)
    return ze / np.sum(ze)


def back_propagate(W, Wd, back, x,lr = 1):
    # print(lr, back, Wd)
    # print((lr * back * Wd))
    # print(x)
    # print(back, Wd, 'back', back * Wd)
    diff_W = - np.dot(np.reshape(x, newshape=[4, 1]), np.reshape(lr * back * Wd, newshape=[1, 3])).T
    #print(diff_W)
    W = W - diff_W
    return W


def counter_right(y, y_f):
    if y.argmax() == y_f.argmax():
        return 1
    else:
        return 0


learning_rate = 0.1
for i in range(500):
    arr_data = np.random.permutation(np.array(data))
    split_data = np.split(arr_data, [int(arr_data.shape[0] * 0.8)])
    train_data = split_data[0]
    test_data = split_data[1]
    diff_W = np.zeros(shape=(3, 4))
    diff_b = np.zeros(shape=(3, ))
    #print(W)
    # lables = train_data[:, -1]
    # y = util.one_hot(lables, np.max(lables) + 1)
    for v in train_data:
        f = v[0:4]
        y = util.one_hot_simple(v[-1], 3)
        y_, Wd = forward(f, W)
        diff = grad_computer(y_, y, f, Wd)
        diff_W += diff[0]
        diff_b += diff[1]
    # print(diff_W)
    # print(diff_W / len(train_data))
    W -= learning_rate * (diff_W / len(train_data))
    b -= learning_rate * (diff_b / len(train_data))
    loss_count, count = loss_cross_entropy(test_data)
    if count > 0.9:
        learning_rate = 0.02
    print(loss_count, count, 'loss')
    # print(right_count)


# a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# b = np.array([1, 2, 3])
# print(a, b)
# print(b * a.T)
