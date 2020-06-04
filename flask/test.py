from keras_vggface.utils import preprocess_input
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, MaxPooling1D, Activation
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from PIL import Image
import numpy as np
from keras_vggface.vggface import VGGFace
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from flask import Flask, redirect, url_for, request, render_template, Response, flash
from gevent.pywsgi import WSGIServer
import cv2
import os
from app import app
from werkzeug.utils import secure_filename
from tensorflow.keras import optimizers
import urllib.request

def get_int_to_name():
    f = open("names100.txt", "r")
    names = []
    for line in f:
        name = line.split("\n")[0]
        names.append(name)
    return dict((number, name) for number, name in enumerate(names))


def get_model(dropout=0.2):
    model = tf.keras.Sequential()

    model.add(Dense(2048, activation='relu', input_shape=(2048,)))
    model.add(Dropout(dropout))
    model.add(Dense(100, activation="softmax"))

    opt = optimizers.Adam()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.load_weights('weights.hdf5')
    return model

model = None
VGG_model = VGGFace(model='resnet50', include_top=False, input_shape=(224,224,3), pooling='avg')
int_to_name = get_int_to_name()
detector = MTCNN()


def extract_face_general(filename, required_size=(224, 224)):
    pixels = pyplot.imread(filename)
    #pixels = load_img('uploads/' + filename, target_size=(224, 224))
    #pixels = img_to_array(image)
    results = detector.detect_faces(pixels)
    faces = []
    pyplot.figure()
    for i in range(len(results)):
        x1, y1, width, height = results[i]['box']
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        image = Image.fromarray(face, 'RGB')
        image = image.resize(required_size)
        pyplot.subplot(1, len(results), i + 1)
        pyplot.imshow(image)
        face_array = np.asarray(image)
        faces.append(face_array)

    #pyplot.show()
    pyplot.savefig('./static/plot.png')
    return np.asarray(faces)


def get_embedding_general(filename, model):
    faces = extract_face_general(filename)
    yhats = []
    for i in range(faces.shape[0]):
        face = faces[i]
        face = np.reshape(face, (1, 224, 224, 3))
        sample = np.asarray(face, 'float32')
        sample = preprocess_input(sample)
        yhat = model.predict(sample)
        yhats.append(yhat.flatten())
    return np.asarray(yhats)


def get_name_general(filename):
    embeddings = get_embedding_general(filename, VGG_model)
    results = []
    for i in range(embeddings.shape[0]):
        embedding = embeddings[i]
        embedding = np.reshape(embedding, (1, -1))
        prediction = model.predict(embedding)
        results.append(int_to_name[np.argmax(prediction[0])])

    return results[0]

def index():
    # Main page
    global model
    global VGG_model
    if model is None:
        model = get_model()


if __name__ == '__main__':
    index()
    print(get_name_general("../test/francesco.jpg"))