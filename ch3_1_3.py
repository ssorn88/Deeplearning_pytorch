import torch

w = torch.tensor(1.0, requires_grad=True)
#requires_grad = True, 파이토치의 Autograd 기능이 자동으로 계산할 때 w에 대한 미분값을 w.grad에 저장

a = w*3
l = a**2
l.backward()
#backward() 미분 함수
print(w.grad)
print('l을 w로 미분한 값은 {}'.format(w.grad))

