import tensorflow as tf
import keras
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss as defined by formula (3)
    
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis = -1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis = -1)
    basic_loss = pos_dist- neg_dist + alpha
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    return loss

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


FRmodel = load_model('MODEL_SAV',custom_objects={'triplet_loss':triplet_loss,'K':K})

    
#@tf.function
#def infer(input_data,model):
#    return model(input_data)
    
def img_to_encoding(image, model):
    image = np.array(image, dtype=np.float32)
    if image.shape[:2] != (96, 96):
        image = tf.image.resize(image, (96, 96))
    img = np.around(np.transpose(image, (2, 0, 1)) / 255.0, decimals=12)
    x_train = np.expand_dims(img, axis=0)
    print(x_train.shape)
    embedding = model.predict(x_train)
    return embedding

def encode(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))  
    if len(faces) == 0:
        raise ValueError("No face detected. Please submit an image with a visible face.")
    elif len(faces) > 1:
        raise ValueError("Must submit an image of a single person.")
    else:
        x, y, w, h = faces[0]
        face = image[y:y + h, x:x + w]
        embedding = img_to_encoding(face, FRmodel)
        return embedding


def detect_verify(frame,database):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))  
    faces_cord=dict()
    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        embedding = img_to_encoding(face, FRmodel)
        #embedding=np.array(embedding['output_0'])
        # Find the closest match in the database
        min_dist = float("inf")
        identity = "Unknown"
        for name, db_enc in database.items():
            dist = np.linalg.norm(embedding-db_enc)
            if dist < min_dist and dist < 0.7:  # Threshold for recognition
                min_dist = dist
                identity = str(name)
        faces_cord[identity] = [int(x),int(y),int(w),int(h)]
    return faces_cord