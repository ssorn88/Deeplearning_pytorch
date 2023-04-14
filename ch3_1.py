import torch

x = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
print(x)
print("size: ", x.size())
print("Shape: ", x.shape)
print("랭크(차원): ", x.ndimension())

# 랭크 늘리기 unsqueeze(텐서명,자리)
# 지정한 차리에 차원값 추가
x = torch.unsqueeze(x,0)
print(x)
print("size: ", x.size())
print("Shape: ", x.shape)
print("랭크(차원): ", x.ndimension())

#랭크 줄이기
#크기가 1인 랭크를 삭제
x = torch.squeeze(x)
print(x)
print("size: ", x.size())
print("Shape: ", x.shape)
print("랭크(차원): ", x.ndimension())

#view()
#랭크 1로 변경, 기존의 텐서 원소 개수를 알고 있어야 함
x = x.view(9)
print(x)
print("size: ", x.size())
print("Shape: ", x.shape)
print("랭크(차원): ", x.ndimension())