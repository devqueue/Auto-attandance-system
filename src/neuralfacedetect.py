#import neuralnetwork
import keras
import utils
import glob
import os
import cv2
import numpy as np
import pickle
from datetime import datetime

# Define variables
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images") 

# Load weights from csv files (which was exported from Openface torch model)
# weights = utils.weights
# weights_dict = utils.load_weights()

# load the model, cascade and trainer
model = keras.models.load_model("./recognizer/neural-network.h5")
face_cascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizer/face-trainner.yml")

# Set layer weights of the model
# for name in weights:
#   if model.get_layer(name) != None:
#     model.get_layer(name).set_weights(weights_dict[name])
#   elif model.get_layer(name) != None:
#     model.get_layer(name).set_weights(weights_dict[name])


def image_to_embedding(image, model):
    #image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA)
    image = cv2.resize(image, (96, 96))
    img = image[..., ::-1]
    img = np.around(np.transpose(img, (0, 1, 2))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding


def recognize_face(face_image, input_embeddings, model):

    embedding = image_to_embedding(face_image, model)

    minimum_distance = 200
    name = None

    # Loop over  names and encodings.
    for (input_name, input_embedding) in input_embeddings.items():

        euclidean_distance = np.linalg.norm(embedding-input_embedding)

        #print('Euclidean distance from %s is %s' % (input_name, euclidean_distance))

        if euclidean_distance < minimum_distance:
            minimum_distance = euclidean_distance
            name = input_name

    if minimum_distance < 0.68:
        return str(name)
    else:
        return None


def create_input_image_embeddings():
    input_embeddings = {}

    for file in glob.glob("images/"):
        person_name = os.path.splitext(os.path.basename(file))[0]
        image_file = cv2.imread(file, 1)
        input_embeddings[person_name] = image_to_embedding(image_file, model)

    return input_embeddings


def markattendance(name):
    with open('Attendance.csv', 'r+') as f:
        dataList = f.readlines()
        nameList = []
        for line in dataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            date = now.strftime('%b %d %Y')
            day = now.strftime('%a')
            time = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name}, {date}, {time}, {day}')
        
input_embeddings = create_input_image_embeddings()
# def recognize_faces_in_cam(input_embeddings):

cv2.namedWindow("Face Recognizer")
cap = cv2.VideoCapture(0)

labels = {"person_name": 1}
with open("pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v: k for k, v in og_labels.items()}

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]  # (ycord_start, ycord_end)
        roi_color = frame[y:y+h, x:x+w]
        #print(w, h)

    	# recognize
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 4 and conf <= 85:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            colortext = (255, 255, 255)
            colorframe = (255, 0, 0)
            stroke = 2
            #bestscale = font_scale(name, w)
            #print(bestscale)
            cv2.rectangle(frame, (x, y), (x + w, y + h), colorframe, stroke)
            cv2.rectangle(frame, (x, y+h), (x+w, y+h+20), colorframe, stroke)
            cv2.rectangle(frame, (x, y+h), (x+w, y+h+20),
                          colorframe, cv2.FILLED)
            cv2.putText(frame, name, (x, y+h+20),
                        font, 0.70, colortext, stroke)
            markattendance(name)

    # Display the resulting frame
    cv2.imshow('Face-detector', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
