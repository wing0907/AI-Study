x = 10
y = 10
w = 0.001
lr = 0.001 # 값에 따라 예측값 핑퐁 때린다
epochs = 1000

for i in range(epochs):
    hypothesis = x * w
    loss = (hypothesis - y) **2 # bias, sigma 고려하지 않음

    print("Loss : ", round(loss, 4), '\tPredict : ', round(hypothesis, 4))
    
    up_predict = x*(w + lr)
    up_loss = (y - up_predict) **2

    down_predict = x*(w - lr)
    down_loss = (y - down_predict) **2

    if(up_loss > down_loss):
        w = w - lr
    else:
        w = w + lr

# 즉 매 에포크마다 w가 0.001씩 증가하고(여기선 𝑤<1이므로 계속 증가),
# 이론적으로 999에포크에서 𝑤=1.000에 도달함.
# 그 시점에선 위/아래 손실이 같아져서 구현상 else로 위로 한 번 갔다가,
# 다음 에포크에 다시 아래로 와서 최적 주변에서 핑퐁하게 됨.