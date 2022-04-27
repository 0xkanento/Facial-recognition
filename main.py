import cv2
import numpy as np

#Eyes detection System
from google.colab.patches import cv2_imshow #I made the script on google colab
img = cv2.imread('') #Path of your image
cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
eyes = cascade.detectMultiScale(img, scaleFactor = 1.2, minNeighbors = 4)

for (x,y,w,h) in eyes:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0, 255, 0),5)


#Face Detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face = face_cascade.detectMultiScale(img,scaleFactor=1.3,minNeighbors=3, minSize=(30, 30))
  
for (x,y,w,h) in face:
  cv2.rectangle(img,(x,y),(x+w,y+h),(0, 255, 0),  5)
#Face Detection
cv2_imshow(img)
