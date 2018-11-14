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