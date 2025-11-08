lst = [1,2,3]
nums = iter(lst)
# print(nums.next())      # python 2.0 문법
print(next(nums))      # 1
print(next(nums))      # 2
print(next(nums))      # 3
# print(next(nums))      # error : StopIteration