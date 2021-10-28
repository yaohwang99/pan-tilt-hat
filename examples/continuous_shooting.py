from time import sleep,strftime,localtime
from vilib import Vilib
from sunfounder_io import PWM,Servo,I2C

import sys
import tty
import termios

# region  read keyboard 
def readchar():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

manual = '''
Press keys on keyboard to record value!
    W: up
    A: left
    S: right
    D: down
    Q: continuous_shooting

    G: Quit
'''
# endregion

# region init
I2C().reset_mcu()
sleep(0.01)

pan = Servo(PWM("P1"))
tilt = Servo(PWM("P0"))
panAngle = 0
tiltAngle = 0

pan.angle(0)
tilt.angle(0)

#endregion init

# region servo control
def limit(x,min,max):
    if x > max:
        return max
    elif x < min:
        return min
    else:
        return x

def servo_control(key):
    global panAngle,tiltAngle       
    if key == 'w':
        tiltAngle -= 1
        tiltAngle = limit(tiltAngle, -90, 90)
        tilt.angle(tiltAngle)
    if key == 's':
        tiltAngle += 1
        tiltAngle = limit(tiltAngle, -90, 90)
        tilt.angle(tiltAngle)
    if key == 'a':
        panAngle += 1
        panAngle = limit(panAngle, -90, 90)
        pan.angle(panAngle)
    if key == 'd':
        panAngle -= 1
        panAngle = limit(panAngle, -90, 90)
        pan.angle(panAngle)

# endregion

# continuous shooting 
def continuous_shooting(path,interval_ms:int=50,number=10):

    for i in range(number):
        Vilib.take_photo(photo_name='%03d'%i,path=path+'/'+strftime("%Y-%m-%d-%H.%M.%S", localtime()))
    print("take_photo: %s"%i)

def main():

    Vilib.camera_start(inverted_flag=True)
    Vilib.display(local=True,web=True)

    path = "/home/pi/picture/continuous_shooting"
  
    print(manual)
    while True:
        key = readchar()
        # servo control
        servo_control(key)
        # take photo
        if key == 'q': 
            print("continuous_shooting .. ")
            continuous_shooting(path,interval_ms=50,number=10)
            print("continuous_shooting done ")
        # esc
        if key == 'g':
            Vilib.camera_close()
            break 

        sleep(0.1)


if __name__ == "__main__":
    main()