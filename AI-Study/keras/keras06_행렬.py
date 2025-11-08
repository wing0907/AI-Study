import numpy as np

x1 = np.array([1,2,3])
print('x1=', x1.shape)    #.shape는 데이터에 대한 구조를 볼 수 있다   # x1= (3,) 벡터

x2 = np.array([[1,2,3]])
print('x2 =', x2.shape)  # x2 = (1, 3)

x3 = np.array([[1,2], [3,4]])
print('x3 =', x3.shape)  # x3 = (2, 2)

x4 = np.array([[1,2], [3,4], [5,6]])
print('x4 =', x4.shape)  # x4 = (3, 2)

x5 = np.array([[[1,2], [3,4], [5,6]]])
print('x5 =', x5.shape)  # x5 = (1, 3, 2)

x6 = np.array([[[1,2], [3,4]], [[5,6],[7,8]]])
print('x6 =', x6.shape)  # x6 = (2, 2, 2)

x7 = np.array([[[[1,2,3,4,5], [6,7,8,9,0]]]])
print('x7 =', x7.shape)  # x7 = (1, 1, 2, 5)

x8 = np.array([[1,2,3], [4,5,6]])
print('x8 =', x8.shape)  # x8 = (2, 3)

x9 = np.array([[[[1]]], [[[2]]]])
print('x9 =', x9.shape)  # x9 = (2, 1, 1, 1)

x10 = np.array([[[[1,2,3,4,5,]]],[[[6,7,8,9,0]]], [[[9,8,7,6,5]]],[[[4,3,2,1,0]]]])
print('x10 =', x10.shape)  # x10 = (4, 1, 1, 5)

