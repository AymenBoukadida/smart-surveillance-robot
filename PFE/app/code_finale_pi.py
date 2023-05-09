

# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
from picamera2 import Picamera2
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import cv2
import numpy as np
import face_recognition
from picamera2 import Picamera2
import os
from datetime import datetime
import time
#from PIL import ImageGrab
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
from email.mime.image import MIMEImage
from email.mime.audio import MIMEAudio

import threading
import pyttsx3
import speech_recognition as sr
import vonage
from playsound import playsound




 
path = '/home/msi/Desktop/images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
    print(classNames)
    
#__________________________________________________________________________________________ 
#save a spoken warning message, "WARNING, Unknown PERSON UP FRONT", as an audio file named "warnning.wav"
#we save it and comment the function , uncomment to change the message spoken
#def audio():

    #engine = pyttsx3.init()
    #voices = engine.getProperty('voices')
    #engine.setProperty('voice', voices[2].id)
    #engine.setProperty('volume', 1.0)
    #engine.setProperty('rate', 150)
    #engine.save_to_file('"Someone in the attached photo is trying to enter the facility. Please review and take necessary action." ', '/home/msi/tflite1/warnning.wav')
    #engine.runAndWait()
    
def alarm():
    engine = pyttsx3.init()


# set audio file path
    audio_file = "/home/msi/Desktop/mixkit-alarm-tone-996.wav"

# play the audio file using playsound
    playsound(audio_file)
    
def markAttendance(name):
    with open('/home/msi/tflite1/attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.write(f'n{name},{dtString}')
    
    

#sending the alert mails
def send_msg(image_path, audio_path):
    message = MIMEMultipart()
    message["from"] = "amineaymenbk12300@gmail.com"
    message["to"] = "AymenBoukadida@proton.me"
    message["subject"] = "Potential issues!!!"
    message.attach(MIMEText("Alert! Our security system has detected a potential issue. Please investigate the situation immediately to ensure the safety and security of the premises."))

    # Attach image to the message
    with open(image_path, 'rb') as f:
        image = MIMEImage(f.read(), _subtype="jpg")
        image.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
        message.attach(image)

    # Attach audio to the message
    with open(audio_path, 'rb') as f:
        audio = MIMEAudio(f.read(), _subtype="wav")
        audio.add_header('Content-Disposition', 'attachment', filename=os.path.basename(audio_path))
        message.attach(audio)

    with smtplib.SMTP(host="smtp.gmail.com", port=587) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.login("amineaymenbk12300@gmail.com", "sdlinunhkitnctwa")
        smtp.send_message(message)
        print("issue sent by mail")

#watching the working directory for the captured image of the unknown person and sends it to the function " send_msg"
def watch_directory(directory):
    while True:
        # Check for both file types before sending the message
        image_path = None
        audio_path = None
        for filename in os.listdir(directory):
            if filename.endswith(".jpg"):
                image_path = os.path.join(directory, filename)

            elif filename.endswith(".wav"):
                audio_path = os.path.join(directory, filename)
        
       
        if image_path or  audio_path:
            send_msg(image_path, audio_path)
            os.remove(image_path)
            
        else:
            time.sleep(2)  # Wait a second before checking again


#2 step authentification              
def check_pass():
    engine = pyttsx3.init()
    
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[2].id)
    engine.setProperty('volume', 1.0)
    engine.setProperty('rate', 150)
    
    
    password = "test"
    
    r = sr.Recognizer()
    
    
    engine.say("for security purposes, provide us with the password")
    engine.runAndWait()
    with sr.Microphone() as source:
                
                audio = r.listen(source)
                try:
                    spoken_text = r.recognize_google(audio)
                    if spoken_text == password:
                       
                        engine.say("Access granted!")
                        engine.runAndWait()
                    else:
                        
                        engine.say("Access denied.")
                        engine.runAndWait()
                except:
                    
                    engine.say("Sorry, I did not understand that.")
                    engine.runAndWait()
                   
def sms_alert():
    client = vonage.Client(key="db5766e1", secret="mTpmOqPC7xBtCpJz")
    sms = vonage.Sms(client)


    responseData = sms.send_message(
    {
        "from": "Vonage APIs",
        "to": "21658995370",
        "text": "an unknown person upfront check ur mail pls",
    }
)

    if responseData["messages"][0]["status"] == "0":
        print("Message sent successfully.")
    else:
        print(f"Message failed with error: {responseData['messages'][0]['error-text']}")
 #________________________________________________________________________  
        
    
    
    
#find the face encodings of each image
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
        print(encode)
    return encodeList


print("Encoding started .....")
encodeListKnown = findEncodings(images)
print('Encoding Complete')
engine = pyttsx3.init()
    
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[2].id)
engine.setProperty('volume', 1.0)
engine.setProperty('rate', 150)
r = sr.Recognizer()

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate


if use_TPU:
    
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME=='edgetpu.tflite' 
                           

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream


picam2 = Picamera2()
picam2.preview_configuration.main.size = (640,480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
already_welcomed=False
prev_result=''

#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1= picam2.capture_array()
    imgS = cv2.resize(frame1,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
    
    # compare the encodings 
    
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        predict=str(faceDis)
        matchIndex = np.argmin(faceDis)
       
    
    
           # labeling the unknown face and the known face 
        if faceDis[matchIndex] < 0.60:
            name = classNames[matchIndex].upper()
            markAttendance(name)
        

            if not already_welcomed:
                # welcome the person
                welcome_text = f"Welcome {name}"
                engine.say(welcome_text)
                engine.runAndWait()
                speech_thread = threading.Thread(target=check_pass)
                speech_thread.start()

            already_welcomed = True
            
        
            
        
        else: 
            name = 'Unknown'
            filename = "Unknown.jpg"
            cv2.imwrite(filename,frame1)
            sms_alert_thread=threading.Thread(target=sms_alert)
            sms_alert_thread.start()
            dir_thread = threading.Thread(target=watch_directory, args=("/home/msi/tflite1/",))
            dir_thread.start()
     
        y1,x2,y2,x1 = faceLoc
        y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(frame1,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(frame1,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
        cv2.putText(frame1,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
       

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects
    detections_text = ""
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            box_scale_factor = 0.6
            ymin = int(max(0,(boxes[i][0] * imH) * box_scale_factor))
            xmin = int(max(0,(boxes[i][1] * imW) * box_scale_factor))
            ymax = int(min(imH,(boxes[i][2] * imH) * box_scale_factor))
            xmax = int(min(imW,(boxes[i][3] * imW) * box_scale_factor))

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            
            
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
            #saving an image of a vol class if detected 
            if object_name == 'vol' and prev_result != 'vol' and scores[0]>0.6 :
                
                filename = "vol_image.jpg"
                cv2.imwrite(filename, frame)
                alarm()
                
                dir_thread = threading.Thread(target=watch_directory, args=("/home/msi/tflite1/",))
                dir_thread.start()
                print(f"Saved {filename}")
            
        # Update previous result
            prev_result = object_name
            # Append detection text
            detections_text += f"{object_name}: {int(scores[i]*100)}%, "

    # Draw detection text overlay on top of frame
    cv2.putText(frame, detections_text[:-2], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(100,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
videostream.stop()
