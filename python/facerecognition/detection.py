import cv2
import facerecognition.utils_cv as utils


def draws_rectangles(img, face_cascade, eye_cascade):
    """Returns the image with the detected faces / eyes surrounded by rectangles
    :param img: the frame to analyse
    :param face_cascade: XML file used for face detection
    :param eye_cascade: XML file used for eye detection
    :return: the modified image"""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Application de l'algorithme de detection des visages
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Affichage du rectangle autour du visage
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # Detection des yeux dans le visage
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            # Affichage deu rectangle autour de l'oeuil
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    return img


def detection_image(path):
    """Displays the specified image and with the detected faces / eyes surrounded by rectangles
    :param path: the path to the image"""

    # Chargement des fichiers XML pour la reconnaissance faciale
    face_cascade = cv2.CascadeClassifier("../../xml/haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("../../xml/haarcascade_eye.xml")

    # Chargement et traitement de l'image
    img = draws_rectangles(utils.load_image(path), face_cascade, eye_cascade)

    # Afficher l'image finale
    utils.display_image(img)


def detection_webcam():
    """Starts a video stream and apply the detection algorithm on each frames"""

    # Capture du flux video
    cap = cv2.VideoCapture(0)

    # Chargement des fichiers XML pour la reconnaissance faciale
    face_cascade = cv2.CascadeClassifier("../../xml/haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("../../xml/haarcascade_eye.xml")

    while True:
        # Capture image par image
        ret, frame = cap.read()

        # Traitement de l'image
        frame = draws_rectangles(frame, face_cascade, eye_cascade)

        # Affichage de l'image
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Fin de la capture du flux video
    cap.release()
    cv2.destroyAllWindows()


detection_webcam()
