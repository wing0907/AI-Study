import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 DEVICE :', DEVICE)



# 1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])


# numpy를 torch tensorflow로 바꿀거임
# x = torch.FloatTensor(x)
# print(x)            # tensor([1., 2., 3.])
# print(x.shape)      # torch.Size([3])
# print(x.size())     # torch.Size([3])

# 행렬이상. 현재 벡터형태. 따라서 reshape ㄱㄱ
x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE)       # unsqueeze 하면 차원을 늘린다. () 안은 순서. 0 하면 ( , 100, 10) 제일 앞쪽이 1이 됨 // 현재 1이기 때문에 (3, 1)이 됨
print(x)
# tensor([[1.],
#         [2.],
#         [3.]])
print(x.shape)          # torch.Size([3, 1])
print(x.size())         # torch.Size([3, 1])
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)
print(y.size())         # torch.Size([3, 1])


######## standard scaling #########
x_mean = torch.mean(x)
x_std = torch.std(x)
x = (x - x_mean) / x_std
###################################
print('스케일링 후 :', x)

# 스케일링 후 : tensor([[-1.],
#         [ 0.],
#         [ 1.]], device='cuda:0')


# 2. 모델구성
# model = Sequential()
# model.add(Dense(1, input_dim=1)) 이게 저 아레 한줄임. 처음 output, 뒤에 input
model = nn.Linear(1, 1).to(DEVICE)             # 하지만 여기서는 앞이 input. 뒤에 output    # y = xw + b

# 3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
criterion = nn.MSELoss()  # loss 정의
# optimizer = optim.Adam(model.parameters(), lr=0.01) # optimizer=adam 정의
optimizer = optim.SGD(model.parameters(), lr=0.1)


### 이부분이 제일 중요함 ### 여기만 이해하면 파이토치는 개쌉꿀임. 디져따리.
def train(model, criterion, optimizer, x, y):
    # model.train()         # 뭔지는 나중에 설명해주신다고함 = 얘는 디폴트임. 명시하지 않아도 먹힌다.
                            # [훈련모드], dropout , batchnormalizaion 적용   
                            # model.train() / optimizer.zero_grad() / loss.backward() / optimizer.step() 이 4가지는 꼭 명시해야함. 1번은 생략 가능
                            
    optimizer.zero_grad()   # 기울기 초기화. ## 파이토치는 zeor_grad 안해주면 기울기가 계속 누적됨
                            # 각 배치마다 기울기를 초기화(0으로)하여, 기울기 누적에 의한 문제 해결
    hypothesis = model(x)   # 모델을 만든다, 가설을 만든다. 이름은 다르게 해도 됨.
                            # y = xw + b // 원래 y 값과 비교되야 함
    
    loss = criterion(hypothesis, y) # loss = mse() = 시그마(y - hypothesis)^2 /n  // 여기까지가 순전파임
                                    # cost, loss 같은 말이라고 하지만.. (이제는 말 할 수 있다 시전하면 헷갈릴 수 있으니 공부하자..)
    # 다시 역전파 들어가야함. // 역전파의 궁극의 목표 = 가중치 갱신 Gradient Descent 가 default. 
    loss.backward()     # 기울기(gradient)값까지만 계산.
    optimizer.step()    # 가중치 갱신
    
    return loss.item()
    
epochs = 800
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch: {}, loss: {}'.format(epoch, loss))  # loss 값이 갱신됨.

# 이제는 말 할 수 있다. 기울기와 가중치는 다르다!!! (추가로 공부하기)
print("============================================================================================")

# 4. 평가, 예측
# loss = model.evaluate(x, y)
def evaluate(model, criterion, x, y):
    model.eval()  # 선생님이 이발하고 오셔서 이발ㄷㄹ하심..ㅂㄷㅂㄷ
                  # [평가모드], dropout , batchnormalization 쓰지 않겠음
    with torch.no_grad():     # gradient 기울기 갱신을 하지 않겠다
        y_predict = model(x)
        loss2 = criterion(y, y_predict) # loss의 최종값
    return loss2.item()

loss2 = evaluate(model, criterion, x, y)
print('최종 loss :', loss2)            # 최종 loss : 0.0

x_pred = (torch.Tensor([[4]]).to(DEVICE) - x_mean) / x_std

result = model(x_pred) # 2차원이니깐 요로케 넣어준다
print('4의 예측값 :', result)           # 4의 예측값 : tensor([[4.]], grad_fn=<AddmmBackward0>)
print('4의 예측값 :', result.item())    # 4의 예측값 : 4.0
