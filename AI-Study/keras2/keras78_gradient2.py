import numpy as np
import matplotlib.pyplot as plt

f = lambda x : x**2 - 4*x + 6

gradient = lambda x : 2*x -4     # f 를 미분한 것

x = -10.0 # 초기값
epochs = 50
learning_rate = 0.1

print('epoch \t x  \t f(x)')  # \t tab 만큼 띄어쓰기
print("{:02d}\t {:6.5f}\t {:6.5f}\t".format(0, x, f(x))) # 현재 1epoch

for i in range(epochs):
    x = x - learning_rate * gradient(x)

    print("{:02d}\t {:6.5f}\t {:6.5f}\t".format(i+1, x, f(x)))


# [분석]
# 대상 함수 𝑓(𝑥)=𝑥2−4𝑥+6
# 2차 함수, 그래프는 U자 모양(볼록 함수)
# 최소값이 하나 존재

# 기울기(gradient) 정의
# 𝑓′(𝑥)=2𝑥−4f ′ (x)=2x−4
# 기울기는 직선 함수, 현재 x값에 따라 양수/음수로 변함
# 목적: 기울기를 이용해 최소점으로 이동

# 초기값과 하이퍼파라미터
# x = -10.0          # 시작점 (왼쪽 멀리)
# epochs = 50        # 반복 횟수
# learning_rate = 0.1  # 학습률

# x: 현재 위치
# learning_rate: 한 번 이동할 때의 "발걸음 크기"
# 너무 크면 발산, 너무 작으면 수렴이 느림

# 경사 하강법 업데이트 식
# x = x - learning_rate * gradient(x)

# 원리:
# # x new​ =x old​ −η⋅f ′ (x old​ )
# η: 학습률 (learning_rate)
# 기울기가 양수면 왼쪽으로, 음수면 오른쪽으로 이동
# 이렇게 반복하면 점점 최소점에 가까워짐

# 수학적으로 최소값 위치 찾기
# f ′ (x)=0 을 풀면:
# 2x−4=0⇒x=2

# 최소값 위치는 𝑥=2
# 최소값은 𝑓(2)=2 2 −4⋅2+6=2

# 실행 흐름
# 처음 x = -10에서 시작
# 매 epoch마다 기울기 계산 → 학습률만큼 이동
# 𝑥 값이 점점 2에 가까워짐
# 50번 반복 후 거의 𝑥=2 에 수렴

# 개선/확장 아이디어 𝑓(𝑥) 를 다른 다변수 함수로 변경 → 다차원 경사 하강법 실습 가능
# matplotlib 로 x의 변화 과정 시각화 가능
# 학습률 변화(Adaptive LR) 적용 가능