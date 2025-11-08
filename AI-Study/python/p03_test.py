
# 2차원 슬라이싱

import numpy as np

# a = np.array([[1,2,3,4,5],
#              [6,7,8,9,0]])
# a = a[:,1]         
# print(a)    #   [2 7]

# a = np.array([[1,2,3,4,5],
#               [6,7,8,9,0],
#               [9,8,7,6,5]])
# a = a[0:2,3]              
# print(a)      # [4]

a = np.array([[1,2,3,4,5],
             [6,7,8,9,0],
             [9,8,7,6,5],
             [4,3,2,1,0]])
a = a[0:3,4]
print(a)    # [5 0 5]


# for = 반복문
#1.
aaa = [1,2,3,4,5]
for i in aaa:       # 리스트 값을 aaa에 넣고, 반복해서 aaa를 i에 넣어줘. 출력했을 때 5개 값이 나온다.
    print(i)
# 1
# 2
# 3
# 4
# 5

#2.
add = 0
for i in range(1, 11):  # range(1, 11-1): 1에서 10까지 들어가 있음
    add = add + i       # 10번 반복해줘
print(add)
# 55

#3.
results=[]              # 리스트의 슬라이싱도 있지만 값을 붙이는 것도 있다. 리스트의 append
for i in aaa:           # append 연결하다 고로 값 + 1 반복으로 해라.
    results.append(i+1) # .list 곧 쓰게 됨
print(results)
# [2, 3, 4, 5, 6]

