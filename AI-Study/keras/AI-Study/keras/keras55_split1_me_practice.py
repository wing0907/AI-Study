import numpy as np

a = np.array(range(1, 21))
timesteps = 19

print(a.shape)  # (20,) 벡터

# def split_x(dataset, timesteps):
#     aaa = []
#     for i in range(len(dataset) - timesteps +1):
#         subset = dataset[i : (i+timesteps)]
#         aaa.append(subset)
#     return np.array(aaa)


def split_xy(dataset, timesteps):
    x, y = [], []
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : i + timesteps]   # ex) [1 2 3 4 5]
        x.append(subset[:-1])                 # 앞 timesteps-1 개 → 입력
        y.append(subset[-1])                  # 마지막 1개 → 타겟
    return np.array(x), np.array(y)

x, y = split_xy(a, timesteps)

print("x:", x)
print("y:", y)
print(x.shape, y.shape) # (11, 9) (11,)

# bbb = split_xy(a, timesteps=timesteps)
# print(bbb)

# [[ 1  2  3  4  5  6  7  8  9 10]
#  [ 2  3  4  5  6  7  8  9 10 11]
#  [ 3  4  5  6  7  8  9 10 11 12]
#  [ 4  5  6  7  8  9 10 11 12 13]
#  [ 5  6  7  8  9 10 11 12 13 14]
#  [ 6  7  8  9 10 11 12 13 14 15]
#  [ 7  8  9 10 11 12 13 14 15 16]
#  [ 8  9 10 11 12 13 14 15 16 17]
#  [ 9 10 11 12 13 14 15 16 17 18]
#  [10 11 12 13 14 15 16 17 18 19]
#  [11 12 13 14 15 16 17 18 19 20]]

# x = bbb[:, :-1]
# y = bbb[:, -1]
# print(x, y)
# print(x.shape, y.shape) # (11, 9) (11,)

