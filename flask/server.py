import torch
import json
from flask import Flask, request

from classification import ClassifierNN

import os
from flask import Flask, flash, request, redirect, send_from_directory
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/tmp/upload_folder/'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model=ClassifierNN()



@app.route('/upload', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        file = request.files['model']
        filename = secure_filename(file.filename)
        print(filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print("Baboo")
        #return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''



@app.route('/predict', methods=['GET','POST'])
def predict():
    file = request.form['model']
    state_dict = torch.load(UPLOAD_FOLDER+file, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    file = request.files['input']
    event = json.load(file)
    event = event["values"]
    tensor = torch.as_tensor(event)
    marshalled_data = tensor.to("cpu")
    with torch.no_grad():
        results = model(marshalled_data)
    return str(results.tolist()[0])


if __name__ == '__main__':
    app.run()
