class Father:  # 괄호없으면 상속받을게 없다는 것
    def __init__(self, name1):
        self.name = name1
        print("Father __init__ 실행됨")
        print(self.name, "발롱도르")
    babo = 4
    
aaa = Father('김치싸대기')

print('----------------------------------------------------------------------')
class Son(Father):      # Father를 상속받을거야
    def __init__(self, name):
        print("Son __init__ 시작")
        super().__init__(name)
        print("Son __init__ 끝")
    cheonje = 5
        
bbb = Son('흥민쏘오오오니')


class Tomato:
    def __init__(self, name):
        self.name = name
        print("Tomato __init__ 실행됨")
        print(self.name, "토마토")
ccc = Tomato('토마토')


class Builder(Tomato, Father):
    def __init__(self, name):
        print('Builder __init__ initiate')
        super().__init__(name)
        print('Builder __init__ done')
ddd = Builder('머쨍이')

    
