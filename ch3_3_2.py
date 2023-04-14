#!/usr/bin/env python
# coding: utf-8

# # 파이토치로 구현하는 신경망
# ## 신경망 모델 구현하기

import torch
import numpy
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import  torch.nn.functional as F

n_dim = 2
# make_blobs, 매개변수 centers: 생성할 클러스터의 수 or 중심, cluster_std: 표준편차
# x_train -> 좌푯값, y_table 에 레이블
x_train, y_train = make_blobs(n_samples=80, n_features=n_dim, centers=[[1, 1], [-1, -1], [1, -1], [-1, 1]],
                              shuffle=True, cluster_std=0.3)
x_test, y_test = make_blobs(n_samples=20, n_features=n_dim, centers=[[1, 1], [-1, -1], [1, -1], [-1, 1]], shuffle=True,
                            cluster_std=0.3)


def label_map(y_, from_, to_):
    y = numpy.copy(y_)
    for f in from_:
        y[y_ == f] = to_
    return y


y_train = label_map(y_train, [0, 1], 0)
y_train = label_map(y_train, [2, 3], 1)
y_test = label_map(y_test, [0, 1], 0)
y_test = label_map(y_test, [2, 3], 1)


def vis_data(x, y=None, c='r'):
    if y is None:
        y = [None] * len(x)
    for x_, y_ in zip(x, y):
        if y_ is None:
            plt.plot(x_[0], x_[1], '*', markerfacecolor='none', markeredgecolor=c)
        else:
            plt.plot(x_[0], x_[1], c + 'o' if y_ == 0 else c + '+')


plt.figure()
vis_data(x_train, y_train, c='r')
plt.show()

x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)


# torch.nn.Module 상속
class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # torch.nm.Linear : 가중치 행렬곱 + 편향 연산, 반환: [1,hidden_size]
        self.linear_1 = torch.nn.Linear(self.input_size, self.hidden_size)
        # ReLU(): 활성화 함수, 입력값<0 -> 0, 입력값>0 -> 입력값
        # Sigmoid(): 활성화 함수 0~1값 반환

        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_tensor):
        linear1 = self.linear_1(input_tensor)
        relu = self.relu(linear1)
        linear2 = self.linear_2(relu)
        output = self.sigmoid(linear2)
        return output


model = NeuralNet(2, 5)  # input_size = 2, hidden_size = 5
learning_rate = 0.03
# BCELLoss() 이진 교차 엔트로피
criterion = torch.nn.BCELoss()
epochs = 2000
# Stochastic Gradient Descent 확률적 선형회귀, 가중치 update
# model.parameter() : 모델 내부의 가중치
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

model.eval()
test_loss_before = criterion(model(x_test).squeeze(), y_test)
print('Before Training, test loss is {}'.format(test_loss_before.item()))

# 오차값이 0.73 이 나왔습니다. 이정도의 오차를 가진 모델은 사실상 분류하는 능력이 없다고 봐도 무방합니다.
# 자, 이제 드디어 인공신경망을 학습시켜 퍼포먼스를 향상시켜 보겠습니다.

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    train_output = model(x_train)
    train_loss = criterion(train_output.squeeze(), y_train)
    if epoch % 100 == 0:
        print('Train loss at {} is {}'.format(epoch, train_loss.item()))
    train_loss.backward()  # 오차함수를 가중치로 미분
    optimizer.step()  # backpropagation


model.eval()
test_loss = criterion(torch.squeeze(model(x_test)), y_test)
print('After Training, test loss is {}'.format(test_loss.item()))

# 학습을 하기 전과 비교했을때 현저하게 줄어든 오차값을 확인 하실 수 있습니다.
# 지금까지 인공신경망을 구현하고 학습시켜 보았습니다.
# 이제 학습된 모델을 .pt 파일로 저장해 보겠습니다.


# state_dict() : 모델 내 가중치들이 {연산이름 : 가중치 텐서와 편향 텐서}
torch.save(model.state_dict(), '../model.pt')
print('state_dict format of the model: {}'.format(model.state_dict()))

# `save()` 를 실행하고 나면 학습된 신경망의 가중치를 내포하는 model.pt 라는 파일이 생성됩니다. 아래 코드처럼 새로운 신경망 객체에 model.pt 속의 가중치값을 입력시키는 것 또한 가능합니다.

new_model = NeuralNet(2, 5)
new_model.load_state_dict(torch.load('../model.pt'))
new_model.eval()
print('벡터 [-1, 1]이 레이블 1을 가질 확률은 {}'.format(new_model(torch.FloatTensor([-1, 1])).item()))

