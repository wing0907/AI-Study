from machine.car import drive
from machine.tv import watch

drive()
watch()

print("="*80)
from machine import car, tv
car.drive()
tv.watch()

print("="*40, "test폴더", "="*40)
from machine.test.car import drive
from machine.test.tv import watch

drive()
watch() 

from machine.test import car, tv
car.drive()
tv.watch() 

from machine import test
test.car.drive()
test.tv.watch()