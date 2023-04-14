import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets


# CUDA(GPU) 사용여부 확인
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print(DEVICE)
# hyper_parameter 설정
EPOCHS = 40
BATCH_SIZE = 64
# dataset
train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./.data',
                          train=True,
                          download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3801,))
                          ])),
    batch_size = BATCH_SIZE, shuffle = True)

test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./.data',
                          train=False,
                          download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3801,))
                          ])),
    batch_size=BATCH_SIZE, shuffle=True)


# 파이토치 nn.Conv2d모듈: 자신을 바로 부를 수 있는 인스턴스
# 입력 x를 받는 함수를 반환
# self.conv1, self.conv2 -> 함수로 취급 될 수 있음


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # nn.Conv2d(입력 채널 수, 출력 채널 수, kernel_size=(n,m)) 커널사이즈 숫자 하나 지정시 정사각형
        # 컨볼루션 계층
        # 10개의 특정 맵 -> 20개의 특징 맵
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size= 5)
        # 드롭아웃
        self.drop = nn.Dropout2d()
        # 일반 신경망
        # 앞 계층의 출력 크기(320) -> 50 -> 분류할 클래스 개수(10)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    # 입력 -> 출력 연결
    def forward(self,x):
        # 맥스 풀링: F.max_poll2d(self, 커널 크기) -> F.relu()
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        # 일반 신경망으로 들어가기 위한 차원 감소(2->1)
        x = x.view(-1,320)
        # 신경망 계층 구성
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# 인스턴스, 최적화 함수
model = CNN().to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# train
def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data,target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 200 == 0:
            print("Train Epoch: {} [{}/{} ({:0f}%]\t Loss:{:6f}"
                  .format(epoch,batch_idx*len(data),
                          len(train_loader.dataset),
                          100 * batch_idx / len(train_loader),
                          loss.item()))


# evaluate
def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct =0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)

            # 배치 오차 합산
            test_loss += F.cross_entropy(output, target,
                                         reduction='sum').item()
            # 가장 높은 값을 가진 인덱스 -> 예측값
            pred = output.max(1, keepdim=True)[1]
            correct +=pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100 * correct / len(test_loader.dataset)
    return test_loss, test_accuracy


for epoch in range(1,EPOCHS+1):
    train(model, train_loader, optimizer, epoch)
    test_loss, test_accuracy = evaluate(model, test_loader)

    print("[{}] Test Loss: {:4f}, Accuracy: {:2f}%"
          .format(epoch, test_loss,test_accuracy))

