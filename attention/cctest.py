# UAS EyeBIS - RealTime Video Capture and Identification
# Modified from instructions from
# https://www.pyimagesearch.com/2017/09/18/real-time-object-detection-with-deep-learning-and-opencv/

from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2

cascade_file = "data/cascade.xml"
video_source = "raw/pos/0.mp4"
# video_source = 0
frame_fov_w = 60
frame_fov_h = 40

# Show Banner
print("""
    Welcome to UAS CCTEST!
""")

# load our serialized model from disk
print("--> Loading model...")
detector = cv2.CascadeClassifier(cascade_file)

print("--> Loading utilities...")


def draw_axes(frame):
    (frame_h, frame_w) = frame.shape[:2]
    cv2.line(frame, (0, frame_h // 2), (frame_w, frame_h // 2), (0, 255, 0), 2)
    cv2.line(frame, (frame_w // 2, 0), (frame_w // 2, frame_h), (0, 255, 0), 2)

    for x in range(0, frame_w, 100):
        angle_x = (x / frame_w - 0.5) * frame_fov_w
        cv2.putText(frame, "{:+.2f}".format(angle_x), (x, frame_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for y in range(0, frame_h, 100):
        angle_y = (y / frame_h - 0.5) * frame_fov_h
        cv2.putText(frame, "{:+.2f}".format(angle_y), (frame_w // 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("--> Starting video stream...")
vs = VideoStream(video_source).start()
time.sleep(2.0)
fps = FPS().start()

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

print("--> Ready!")

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream
    frame = vs.read()
    (frame_h, frame_w) = frame.shape[:2]

    draw_axes(frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    objects = detector.detectMultiScale(
        gray,
        scaleFactor=1.01,
        minNeighbors=5,
        minSize=(24, 24),
        flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in objects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        center_x = x + w // 2
        center_y = y + h // 2
        # cv2.circle(frame, (center_x, center_y), 2, (0, 0, 255), -1)
        angle_x = (center_x / frame_w - 0.5) * frame_fov_w
        angle_y = (center_y / frame_h - 0.5) * frame_fov_h
        cv2.putText(frame, "{:+.2f}, {:+.2f}".format(angle_x, angle_y), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()
    fps.stop()
    print("[INFO] Elapsed: {}s FPS: {:.2f}".format(fps.elapsed(), fps.fps()))

# stop the timer and display FPS information
fps.stop()

# do a bit of cleanup
vs.stop()
cv2.destroyAllWindows()
