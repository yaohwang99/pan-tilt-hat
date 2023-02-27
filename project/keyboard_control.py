
from servo import Servo
import cv2
pan = Servo(pin=13) # pan_servo_pin (BCM)
tilt = Servo(pin=12)

pan.set_angle(6)
tilt.set_angle(-60)
while True:
    c = input()
    if c == "w":
        tilt.decrease_angle()
    elif c == "s":
        tilt.increase_angle()
    elif c == "a":
        pan.increase_angle()
    elif c == "d":
        pan.decrease_angle()
    elif c == "g":
        print("pan angle: " + str(pan.get_angle()))
        print("tilt angle: " + str(tilt.get_angle()))
