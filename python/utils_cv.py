import cv2


def load_and_display_image(path):
    """Loads and display the specified image
    :param path: the path to the file"""

    img = cv2.imread(path, 0)  # chargement de l'image

    # Création de la fenetre et affichage
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
