from imutils.video import VideoStream
from flask import Response
from flask import Flask, request
from flask import render_template
from picamera.array import PiRGBArray
from picamera import PiCamera
import threading
import argparse
import datetime
import imutils
import time
import cv2
import pytz
from pytz import timezone
from servo import Servo
from time import sleep
import numpy as np
pan = Servo(pin=13) # pan_servo_pin (BCM)
tilt = Servo(pin=12)
# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)
# initialize the video stream and allow the camera sensor to
# warmup
# vs = VideoStream(usePiCamera=1).start()
vs = VideoStream(src=0).start()
time.sleep(2.0)

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

def move_servo():
    global pan, tilt
    pan_mid = 6
    tilt_mid = -60
    offset = 0
    d = 1
    while True:
        pan.set_angle(pan_mid + offset)
        if offset == 30:
            d = -1
        elif offset == -30:
            d = 1
        offset += d
        sleep(0.1)
def read_frame():
    global vs, outputFrame, lock
    frame = vs.read()
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 100, 20])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([160,100,20])
    upper2 = np.array([179,255,255])
    lower_mask = cv2.inRange(hsv, lower1, upper1)
    upper_mask = cv2.inRange(hsv, lower2, upper2)
    mask = lower_mask + upper_mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    rx = 300
    ry = 300
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            rx = x
            ry = y
    # grab the current timestamp and draw it on the frame
    timestamp = datetime.datetime.now(tz=pytz.utc)
    timestamp = timestamp.astimezone(timezone('US/Pacific'))
    cv2.putText(frame, timestamp.strftime(
        "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    with lock:
        outputFrame = frame.copy()
    return 300 - rx, ry - 300
def detect_motion():
    # grab global references to the video stream, output frame, and
    # lock variables
    # Define PID constants
    # KP = 0.01
    # KI = 0.01
    # KD = 0.005
    # Initialize variables
    x_error_sum = 0
    x_last_error = 0
    y_error_sum = 0
    y_last_error = 0
    last_time = time.time()
    while True:
        current_time = time.time()
        x_error, y_error = read_frame()
        time_diff = current_time - last_time
        x_error_sum += x_error * time_diff
        y_error_sum += y_error * time_diff
        tilt_angle = tilt.get_angle() + controller(0.015, 0, 0.0005,y_error, y_last_error, y_error_sum, time_diff)
        tilt.set_angle(tilt_angle)
        pan_angle = pan.get_angle() + controller(0.015, 0, 0.0005,x_error, x_last_error, x_error_sum, time_diff)
        pan.set_angle(pan_angle)
        x_last_error = x_error
        y_last_error = y_error
        last_time = current_time
        time.sleep(0.01)
def controller(kp, ki, kd, error, last_error, error_sum, time_diff):
    error_diff = error - last_error
    derivative = error_diff / time_diff
    error_sum += error * time_diff
    integral = error_sum
    output = kp * error + ki * integral + kd * derivative
    return output
        

def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock
	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue
			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
			# ensure the frame was successfully encoded
			if not flag:
				continue
		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')
    

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
        help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
        help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
        help="# of frames used to construct the background model")
    args = vars(ap.parse_args())
    # start a thread that will perform motion detection
    t = threading.Thread(target=detect_motion)
    t.daemon = True
    t.start()
    # servo_worker = threading.Thread(target=move_servo)
    # servo_worker.daemon = True
    # servo_worker.start()
    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
        threaded=True, use_reloader=False)
# release the video stream pointer
vs.stop()