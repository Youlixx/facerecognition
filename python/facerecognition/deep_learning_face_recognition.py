import face_recognition
import cv2
import numpy

from imutils import paths


def create_data(database_path, detection_method="hog"):
    """Generates the 128-d vectors of the faces of the database
    :param database_path: the path to the database folder
    :param detection_method: the detection method (hog or cnn but cnn is slower)
    :return: the data"""

    image_paths = list(paths.list_images(database_path))

    known_encodings = []
    known_names = []

    for i, image_path in enumerate(image_paths):
        image_name = image_path.replace(database_path + "\\", "")
        name = image_name.split("_")[0]

        print("PROCESSING ", i+1, "/", len(image_paths), image_name)

        image = cv2.imread(image_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb, model=detection_method)
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(name)

    data = {"encodings": known_encodings, "names": known_names}

    return data


def save_data(data, path):
    file = open(path, "w")

    lines = []

    for k in range(len(data["encodings"])):
        line = data["names"][k] + ":"

        for j in range(len(data["encodings"][k])):
            if j != 0:
                line += ";"

            line += str(data["encodings"][k][j])

        lines.append(line + "\n")

    file.writelines(lines)
    file.flush()
    file.close()


def load_data(path):
    file = open(path, "r")

    encodings = []
    names = []

    line = file.readline()

    while line != "":
        names.append(line.split(":")[0])

        encoding = numpy.zeros((128, ))
        values = line.split(":")[1].split(";")

        for k in range(128):
            encoding[k] = float(values[k])

        encodings.append(encoding)

        line = file.readline()

    data = {"encodings": encodings, "names": names}

    return data


def detection(data, image_path, detection_method="hog"):
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb, model=detection_method)
    encodings = face_recognition.face_encodings(rgb, boxes)

    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "UNKNOWN"

        if True in matches:
            matched_id = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matched_id:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)

        names.append(name)

    faces = {"boxes": boxes, "names": names}

    return faces


def draw_boxes(data, image_path, detection_method="hog"):
    faces = detection(data, image_path, detection_method)

    boxes = faces["boxes"]
    names = faces["names"]

    image = cv2.imread(image_path)

    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

        y = top - 15 if top - 15 > 15 else top + 15

        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # show the output image
    cv2.imshow("Image", image)
    cv2.waitKey(0)


save_data(create_data("../../data/database", detection_method="cnn"), "../../data/database.data")

#draw_boxes(load_data("../../data/database.data"), "../../data/nico.jpg")

