import machine.car, machine.tv

# car.drive() # 이렇게는 인식이 안된다.
# drive() # 이것도 인식 안된다.

machine.car.drive() # 이렇게 해야 인식이 된다.
machine.tv.watch()

