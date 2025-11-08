#  실습 : Son에서 babo를 출력해라

class Father:  # 괄호없으면 상속받을게 없다는 것
    def __init__(self, name):
        self.name = name
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
        print("가즈아아:", super().babo + self.cheonje)
        print("안뇨옹?:", self.babo + self.cheonje)
    cheonje = 5
    
bbb = Son('흥민쏘오오오니')

print('참전용사만세', Father.babo+ Son.cheonje)
# 가즈아아: 9
# 안뇨옹?: 9
# 참전용사만세 9   

