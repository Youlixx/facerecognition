import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


#def load_and_display_image(path):
'''def L(path):
    """Loads and display the specified image
    :param path: the path to the file"""

    img = cv2.imread(path, 1)  # chargement de l'image


    # Création de la fenetre et affichage
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

#Détection de visages


face_cascade = cv.CascadeClassifier('C:/Users/gui-f/dossierJ0/xml/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('C:/Users/gui-f/dossierJ0/xml/haarcascade_eye.xml')
img = cv.imread('C:/Users/gui-f/dossierJ0/data/test.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
cv.imshow('img',img)
cv.imwrite('guillaumeF.png',img)
cv.waitKey(0)
cv.destroyAllWindows()

#détection de visages en continu:
'''face_cascade = cv.CascadeClassifier('C:/Users/gui-f/dossierJ0/xml/haarcascade_frontalface_default.xml')

eye_cascade = cv.CascadeClassifier('C:/Users/gui-f/dossierJ0/xml/haarcascade_eye.xml')

   #Webcam capture
video_capture = cv.VideoCapture(0)
while True :
    ret, frame = video_capture.read()
    frame = cv.flip(frame, 1)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)


    for (x,y,w,h) in faces:
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv.imshow('img',frame)
    if cv.waitKey(1) == 27:
        break  # esc to quit
cv.destroyAllWindows()
video_capture.release()'''
