from flask import Flask, render_template, request, redirect, jsonify, send_file
# from test import TextRecognition
from NNbackend import recognizer
import os, io ,sys
import base64
from PIL import Image
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = './static/uploads/'


@app.route('/')
def index():
    return render_template("miniProj.html")


@app.route('/recognize', methods=['GET', 'POST'])
def reco():
    file = request.files['file']
    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)
    Map = recognizer(path)
    return jsonify({'status':'succces'})



@app.route('/test' , methods=['GET','POST'])
def test():
	print("log: got at test" , file=sys.stderr)
	return jsonify({'status':'succces'})


@app.route('/download')
def downloadFile ():
    #For windows you need to use drive name [ex: F:/Example.pdf]
    path = "data.txt"
    return send_file(path, as_attachment=True)

@app.route('/report')
def downloadReport ():
    #For windows you need to use drive name [ex: F:/Example.pdf]
    path = "SHIPRADAR_Mini Project Report.pdf"
    return send_file(path, as_attachment=True)

    
if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=True)

