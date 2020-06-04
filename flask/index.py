from keras_vggface.utils import preprocess_input
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from PIL import Image
import numpy as np
import time
from keras_vggface.vggface import VGGFace
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import glob
from flask import Flask, redirect, url_for, request, render_template, Response, flash, after_this_request
from gevent.pywsgi import WSGIServer
from tensorflow.keras.models import load_model
import os
from app import app
from werkzeug.utils import secure_filename
from tensorflow.keras import optimizers

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


model = get_model()
VGG_model = VGGFace(model='resnet50', include_top=False, input_shape=(224,224,3), pooling='avg')
int_to_name = get_int_to_name()
detector = MTCNN()
loaded = False

def extract_face_general(filename, required_size=(224, 224)):
    pixels = pyplot.imread(os.path.join(app.config['UPLOAD_FOLDER'],filename))
    results = detector.detect_faces(pixels)
    faces = []
    pyplot.figure()
    if len(results) > 1:
        nb_of_faces = "Detected " + str(len(results)) + " faces"
        flash(nb_of_faces)
    elif len(results) == 1:
        nb_of_faces = "Detected 1 face"
        flash(nb_of_faces)
    else:
        nb_of_faces = "No faces found, please upload another picture"
        flash(nb_of_faces)
        return -1
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
    save_file = filename.split('.')[0]
    pyplot.savefig('./static/plots/plot_' + save_file + '.png', bbox_inches='tight')
    flash('./static/plots/plot_' + save_file + '.png')
    return np.asarray(faces)


def get_embedding_general(filename, model):
    faces = extract_face_general(filename)
    if faces.all() == -1:
        return -1
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
    global model
    global VGG_model
    embeddings = get_embedding_general(filename, VGG_model)
    if embeddings.all() == -1:
        return
    results = []
    for i in range(embeddings.shape[0]):
        embedding = embeddings[i]
        embedding = np.reshape(embedding, (1, -1))
        prediction = model.predict(embedding)
        results.append(int_to_name[np.argmax(prediction[0])])

    str = ', '.join(results)
    flash(str)
    return results


@app.route('/', methods=['GET'])
def index():
    # Main page
    global model
    global VGG_model
    global loaded
    if model is None:
        model = get_model()

    full_dir = glob.glob("./static/plots/plot_*.png")
    if loaded == True:
        loaded = False
        if(len(full_dir) > 1):
            for i in range(len(full_dir)):
                os.remove(full_dir[i])
    else:
        for i in range(len(full_dir)):
            os.remove(full_dir[i])

    return render_template('index.html')

# @app.route('/plot.png')
# def plot_png():
#     fig = './static/plots/plot_' + save_file + '.png'
#     output = io.BytesIO()
#     FigureCanvas(fig).print_png(output)
#     return Response(output.getvalue(), mimetype='image/png')

@app.route('/', methods=['POST'])
def submit_file():
    global loaded
    loaded = True
    full_dir = glob.glob("./static/plots/plot_*.png")
    for i in range(len(full_dir)):
        os.remove(full_dir[i])
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            get_name_general(filename)
            # save_file = filename.split('.')[0]
            # flash('./static/plots/plot_' + save_file + '.png')

            @after_this_request
            def remove_file(response):
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'],filename))
                #os.remove('./static/plots/plot_' + save_file + '.png')
                return response

            return redirect('/')


if __name__ == '__main__':
    full_dir = glob.glob("./static/plots/plot_*.png")
    for i in range(len(full_dir)):
        os.remove(full_dir[i])
    #app.run(debug=True, threaded=False)

    #Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    print("Server started")
    http_server.serve_forever()