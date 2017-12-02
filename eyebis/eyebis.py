# UAS EyeBIS - RealTime Video Capture and Identification
# Modified from instructions from
# https://www.pyimagesearch.com/2017/09/18/real-time-object-detection-with-deep-learning-and-opencv/

from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2

# Show Banner
print("""
    Welcome to UAS EyeBIS!
    It's a bird pun in "ibis".
""")

# Model-Specific Configurations
models = {

    # SSD Model using MobileNet
    # GitHub Citation:
    # https://github.com/chuanqi305/MobileNet-SSD
    'MobileNetSSD_deploy': {
        'prefix': "models/MobileNetSSD/MobileNetSSD_deploy",
        'bgr_sub': 127.5,
        'normalize': lambda x: x[0, 0],
        'classes': ["background", "aeroplane", "bicycle", "bird", "boat",
                    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                    "sofa", "train", "tvmonitor"],
    },

    # SSD300_VGG
    # GitHub Citation: https://github.com/weiliu89/caffe/tree/ssd#models
    # Very slow!!
    'SSD300_VGG': {
        'prefix': "models/SSD300_VGG/VGG_ILSVRC2016_SSD_300x300_iter_440000",
        'bgr_sub': [103.939, 116.779, 123.68],
        'normalize': lambda x: x[0, 0],
        'classes': ["background", "aeroplane", "bicycle", "bird", "boat",
                    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                    "sofa", "train", "tvmonitor"],
    },
}

colors = {}

# PARAMETERS
parameters = {
    'model': "MobileNetSSD_deploy",
    'min_confidence': 0.2,
    'video_source': 0,  # Use 0 for webcam
    # 'video_source': "aerial.mp4",
    # 'reshape': (640, 360),
    'reshape': (300, 300),
    'targets': ['car', 'person']
}

# load our serialized model from disk
print("--> Loading model...")
model = models[parameters['model']]
net = cv2.dnn.readNetFromCaffe(model['prefix'] + ".prototxt", model['prefix'] + ".caffemodel")

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("--> Starting video stream...")
vs = VideoStream(src=parameters['video_source']).start()
time.sleep(2.0)
fps = FPS().start()

print("--> Ready!")

cv2.namedWindow('VISION', cv2.WINDOW_NORMAL)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, parameters['reshape']),
                                 0.007843, parameters['reshape'], model['bgr_sub'])

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()
    detections = model['normalize'](detections)

    # loop over the detections
    for i in range(detections.shape[0]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > parameters['min_confidence']:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[i, 1])
            box = detections[i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            name = model['classes'][idx] if idx < len(model['classes']) else "unknown ({})".format(idx)

            if name not in parameters['targets']:
                continue

            centroid = ((startX + endX) / 2, (startY + endY) / 2)
            label = "{}: {:.2f}% @({}, {})".format(name,
                                                   confidence * 100, *centroid)

            # draw the prediction on the frame
            if not idx in colors:
                colors[idx] = np.random.uniform(0, 255, size=3)
            color = colors[idx]

            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # show the output frame
    cv2.imshow("VISION", frame)
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
