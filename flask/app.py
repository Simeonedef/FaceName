from flask import Flask

app = Flask(__name__)
UPLOAD_FOLDER = "./uploads"
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMPLATES_AUTO_RELOAD'] = True