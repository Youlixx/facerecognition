import cv2


def load_image(path):
    """Loads the specified image from the file
    :param path: the path to the file"""

    return cv2.imread(path, cv2.IMREAD_COLOR)


def display_image(image):
    """Displays the image
    :param image: the image array representation"""

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def take_photo(path):
    """Takes a photo using the webcam after the user pressed q
    :param path: the path to the image"""

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite(path, frame)

            break

    cap.release()
    cv2.destroyAllWindows()
