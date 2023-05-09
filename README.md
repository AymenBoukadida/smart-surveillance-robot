# Un-robot-daccuiel-avec-reconnaissance-faciale-vocale-et-detection-dobjet:

DESCRIPTION:
The robot would consist of a Raspberry Pi 4 computer connected to a camera module for image and video capture, a microphone for audio input, a speaker for audio output, and sensors for detecting objects. The robot could be mounted on a wheeled or stationary base, depending on the desired mobility.

The software running on the Raspberry Pi 4 would use OpenCV for image processing and object detection. It would also use a pre-trained machine learning model for facial recognition and speech recognition. When an unknown face is detected, the robot would take a photo and send an email alarm to the user to notify them of the event. When a "vol"  object is detected, the robot would sound an audio alarm through its speaker and end a mail along with the picture of the object 

PACKGES USED  :
opencv-python ,
tensorflow (for windows),
tf-lite (for pi),
numpy,
speech-recognition,
face recognition,
pyttsx3 ,
playsound,
threading,
vonage(sms api),
csv,
smtplib(mail api),
theading ,
need to install object detection api for the windows version and cudda (used ssd mobilenet v2 fpnlite 320x320 from tensorflow),

