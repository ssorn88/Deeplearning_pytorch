
import torch
import pickle
import matplotlib.pyplot as plt


shp_original_img = (100, 100)
broken_image = torch.FloatTensor(pickle.load(open('../broken_image_t.p', 'rb'), encoding='latin1'))


plt.imshow(broken_image.view(100,100))
plt.show()


def weird_function(x, n_iter=5):
    h = x
    filt = torch.tensor([-1./3, 1./3, -1./3])
    for i in range(n_iter):
        zero_tensor = torch.tensor([1.0*0])
        h_l = torch.cat( (zero_tensor, h[:-1]), 0)
        h_r = torch.cat((h[1:], zero_tensor), 0 )
        h = filt[0] * h + filt[2] * h_l + filt[1] * h_r
        if i % 2 == 0:
            h = torch.cat( (h[h.shape[0]//2:],h[:h.shape[0]//2]), 0  )
    return h


#loss_function
def distance_loss(hypothesis, broken_image):
    return torch.dist(hypothesis, broken_image)


random_tensor = torch.randn(10000, dtype = torch.float)

# 학습률, 학습을 얼마나 급하게 진행하는가
lr = 0.8
for i in range(0,20000):
    random_tensor.requires_grad_(True)
    hypothesis = weird_function(random_tensor)
    loss = distance_loss(hypothesis, broken_image)
    # loss를 random_tensor로 미분, 갑은 random_tensor.grad에 저장
    loss.backward()
    # 자동 기울기 계산 비활성화
    with torch.no_grad():
        # random_tensor.grad: loss.backward()함수에서 계산한 loss의 기울기, loss가 최대값이 되는 방향
        random_tensor = random_tensor - lr*random_tensor.grad
    if i % 1000 == 0:
        print('Loss at {} = {}'.format(i, loss.item()))


plt.imshow(random_tensor.view(100,100).data)
plt.show()

