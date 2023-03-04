################# Loading libraries and frameworks #################
from flask import Flask, render_template, Response
import cv2
import os
import json
import base64
from flask import send_file
from flask import request
import numpy as np 
import cv2,joblib
from pathlib import Path
import tempfile
from werkzeug.utils import secure_filename
from imutils.object_detection import non_max_suppression
import imutils
from skimage.feature import hog
from skimage import color
from skimage.transform import pyramid_gaussian
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2


app = Flask(__name__)


# derive the paths to the YOLO weights and model configuration

n_faces = 0

#################### Get video ####################
## Video path. By default, we get the webcam ## 
video_path=0
###################################################
camera = cv2.VideoCapture(video_path)
###################################################

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


# Function which allows us to detect the face #
def detectCamera():  
    CLASSES = ["person"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

    # initialize the video stream, allow the cammera sensor to warmup,
    # and initialize the FPS counter
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    fps = FPS().start()
    while True:
        counter = 0
        
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
            0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()

	# loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > 0.2:
                idx = int(detections[0, 0, i, 1])
                if idx != 15:
                    continue

                counter = counter + 1
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = 0
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx],
                    confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
        
        cv2.putText(frame, str(counter), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2) # creates green color text with text size of 0.5 & thickness size of 2

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 
        #######################################################################################
############################################


def detectImage():
    CLASSES = ["person"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    counter = 0
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

    image = request.files['file'].read()
    # decode image
    nparr = np.fromstring(image, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
        0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and
        # predictions
    net.setInput(blob)
    detections = net.forward()
    
    for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
        confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
        if confidence > 0.2:
            
            idx = int(detections[0, 0, i, 1])
            if idx != 15:
                continue

            counter = counter + 1
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
            idx = 0
            
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

                # draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx],
                confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    cv2.putText(frame, str(counter), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2) # creates green color text with text size of 0.5 & thickness size of 2

    ret, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer)


def detectVideo():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join('UPLOAD_FOLDER', filename))

    CLASSES = ["person"] 
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

    # initialize the video stream, allow the cammera sensor to warmup,
    # and initialize the FPS counter
    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture('UPLOAD_FOLDER/'+filename)
    
    fps = FPS().start()
    # We need to set resolutions.
    # so, convert them from float to integer.
    frame_width = int(vs.get(3))
    frame_height = int(vs.get(4))
    
    size = (frame_width, frame_height)
    
    # Below VideoWriter object will create
    # a frame of above defined The output 
    # is stored in 'filename.avi' file.
    out = cv2.VideoWriter('filename.avi', 
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            10, size)
    window_name = 'frame'
    delay = 1

# Read until video is completed
    while(vs.isOpened()):
        
        counter = 0
        ret, frame = vs.read()
        if ret == True:
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                0.007843, (300, 300), 127.5)

            # pass the blob through the network and obtain the detections and
            # predictions
            net.setInput(blob)
            detections = net.forward()
            for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # the prediction
                confidence = detections[0, 0, i, 2]
                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence > 0.2:
                    idx = int(detections[0, 0, i, 1])
                    if idx != 15:
                        continue

                    counter = counter + 1
                    # extract the index of the class label from the
                    # `detections`, then compute the (x, y)-coordinates of
                    # the bounding box for the object
                    idx = 0
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # draw the prediction on the frame
                    label = "{}: {:.2f}%".format(CLASSES[idx],
                        confidence * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                        COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            
            cv2.putText(frame, str(counter), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2) # creates green color text with text size of 0.5 & thickness size of 2
            out.write(frame)            
        # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                vs.release()
                out.release()
                break
  
    # Break the loop
        else:
            vs.release()
            out.release()
            break
    vs.release()
    out.release()
    return 'filename.avi'
        #######################################################################################
############################################


######################## Routing to the face detection function ########################
@app.route('/video_feed')
def video_feed():
    return Response(detectCamera(), mimetype='multipart/x-mixed-replace; boundary=frame')
##########################################################################################

################# Main page #################
@app.route('/')
def index():
    return render_template('index.html')
#############################################
@app.route('/detect-by-camera')
def camera():
    return render_template('camera.html')

@app.route('/detect-by-image')
def image():
    return render_template('image.html')

@app.route('/detect-by-image-upload',methods=['POST'])
def imageUp():
    return Response(detectImage())

@app.route('/detect-by-video')
def video():
    return render_template('video.html')

@app.route('/detect-by-video-upload',methods=['POST'])
def videoUp():
    return Response(detectVideo())

@app.route('/uploads/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    return send_file(filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
