import numpy as np

a = np.array(range(1, 11))
timesteps = 5

print(a.shape)  # (10,) 벡터

def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps +1):
        subset = dataset[i : (i+timesteps)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, timesteps=timesteps)
print(bbb)

# 1부터 10까지의 데이터가 6행 5열로 바뀜. 원래 시계열 데이터인데 timesteps 크기로 잘라버림.
# x와 y를 정의해줘야 하는데 
# [[ 1  2  3  4  5]
#  [ 2  3  4  5  6]
#  [ 3  4  5  6  7]
#  [ 4  5  6  7  8]
#  [ 5  6  7  8  9]
#  [ 6  7  8  9 10]]
#  1,2,3 까지 x로 잡고 y를 4,5 로 정의해주면 된다.

x = bbb[:, :-1]     
y = bbb[:, -1]
print(x, y)
print(x.shape, y.shape) # (6, 4) (6,)

