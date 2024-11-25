import csv
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import time
#from PIL import ImageGrab
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
from email.mime.image import MIMEImage
from email.mime.audio import MIMEAudio
import tensorflow as tf
import threading
import pyttsx3
import speech_recognition as sr
import vonage
from playsound import playsound




 
path = './venv/images'
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
def audio():

    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[2].id)
    engine.setProperty('volume', 1.0)
    engine.setProperty('rate', 150)
    engine.save_to_file('WARNNING, Unknown PERSON UP FRONT ', './venv/warnning.wav')
    engine.runAndWait()
    
    
def markAttendance(name):
    with open('./venv/Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'n{name},{dtString}')
    

#sending the alert mails
def send_msg(image_path, audio_path):
    message = MIMEMultipart()
    message["from"] = "from email"
    message["to"] = "to email"
    message["subject"] = "Unknown Person Detected"
    message.attach(MIMEText("This is the picture of the Unknown person up front!!"))

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
        smtp.login("from email", "password")
        smtp.send_message(message)
        print("Sent the image and audio of the Unknown")

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
        
       
        if image_path and audio_path:
            send_msg(image_path, audio_path)
            os.remove(image_path)
            os.remove(audio_path)
        else:
            time.sleep(3)  # Wait a second before checking again


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
    client = vonage.Client(key="db5766e1", secret="secret")
    sms = vonage.Sms(client)


    responseData = sms.send_message(
    {
        "from": "Vonage APIs",
        "to": "numero",
        "text": "This is your security system, Please Check your E-mail for intrusion information NOW",
    }
)

    if responseData["messages"][0]["status"] == "0":
        print("Message sent successfully.")
    else:
        print(f"Message failed with error: {responseData['messages'][0]['error-text']}")
        
        
def alarm():
    engine = pyttsx3.init()


# set audio file path
    audio_file = "./venv/mixkit-alarm-tone-996.wav"

# play the audio file using playsound
    playsound(audio_file)
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
    

    

cap = cv2.VideoCapture(0)
pTime = 0
cap.set(cv2.CAP_PROP_FPS, 100)
cap.set(3,640)
cap.set(4,480)
already_welcomed = False
filename = 'attendance.csv'
if not os.path.exists(filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Name', 'Time'])
while True:

    counter = 0
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    
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
            
            if not already_welcomed:
                # welcome the person
                welcome_text = f"Welcome {name}"
                engine.say(welcome_text)
                engine.runAndWait()
                speech_thread = threading.Thread(target=check_pass)
                
                speech_thread.start()

            already_welcomed = True
            with open(filename, 'r+', newline='') as file:
                reader = csv.reader(file)
                writer = csv.writer(file)
                rows = list(reader)
                for row in rows:
                    if row[0] == name:
                        row[1] = time.strftime('%Y-%m-%d %H:%M:%S')
                        break
                else:
                    writer.writerow([name, time.strftime('%Y-%m-%d %H:%M:%S')])
        
            
        
        else: 
            name = 'Unknown'
            filename = "./venv/Unknown.jpg"
            cv2.imwrite(filename,img)
            
            
            audio_thread = threading.Thread(target=audio)
            audio_thread.start()

            dir_thread = threading.Thread(target=watch_directory, args=("./venv",))
            dir_thread.start()
     
        y1,x2,y2,x1 = faceLoc
        y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
        cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        
    
    cv2.imshow('Webcam',img)
    
    k = cv2.waitKey(5) & 0xff
    if k == 27: # press 'ESC' to quit
        break
    
cap.release()
cv2.destroyAllWindows()





