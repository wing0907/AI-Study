list = ['a', 'b', 'c', 'd', 5]
print(list)

for i in list:
    print(i)
print('======================')
for index, value in enumerate(list):
    print(index, value)

# ['a', 'b', 'c', 'd', 5]
# a
# b
# c
# d
# 5
# ======================
# 0 a
# 1 b
# 2 c
# 3 d
# 4 5
