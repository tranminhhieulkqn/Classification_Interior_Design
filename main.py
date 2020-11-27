import os

from flask import Flask, jsonify, render_template, flash, request, redirect
from werkzeug.utils import secure_filename
from PIL import Image
from django.core.files.storage import default_storage
from google.cloud import storage
from google.cloud.storage import Blob
from source.ModelGeneral import ModelGeneral

#UPLOAD_FOLDER = 'static/uploaded_images/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['BUCKET'] = 'imguploaded'
app.config['UPLOAD_FOLDER'] = '/tmp'

if __name__ == 'app':
    my_model = ModelGeneral()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_picture(form_picture,name):
    picture_fn = secure_filename(name + '.jpg')
    picture_path = os.path.join(app.config['UPLOAD_FOLDER'], picture_fn)
    output_size = (1000,1000)
    i = Image.open(form_picture)
    i.thumbnail(output_size)
    i.save(picture_path)

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(app.config['BUCKET'])
    blob = bucket.blob('static/uploaded_images/'+ picture_fn)
    blob.upload_from_filename(picture_path)

    return picture_path

@app.route('/', methods=['GET', 'POST'])
def home_page():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            #filename = secure_filename(file.filename)
            #image_path = app.config['UPLOAD_FOLDER'] + filename
            #file.save(image_path)

            #filename = save_picture(form.image.data,file.filename)
            filename = secure_filename(file.filename)
            image_path = save_picture(filename,file.filename)
            #file.save(image_path)

            try:
                labels = my_model.get_lables()
                xception = my_model.prediction(model="Xception", image_path=image_path)
                inceptionv3 = my_model.prediction(model="InceptionV3", image_path=image_path)
                densenet201 = my_model.prediction(model="DenseNet201", image_path=image_path)
            except:
                return jsonify({
                    "success": False,
                    "message": "File not exist!"
                }), 404

            os.remove(image_path)

            return jsonify({
                "success": True,
                "message": "Predicted Results",
                "Labels": labels.tolist(),
                "Xception": [round(num, 2) for num in xception[0].tolist()],
                "Xception Predicted": xception[1],
                "InceptionV3": [round(num, 2) for num in inceptionv3[0].tolist()],
                "InceptionV3 Predicted": inceptionv3[1],
                "DenseNet201": [round(num, 2) for num in densenet201[0].tolist()],
                "DenseNet201 Predicted": densenet201[1]
            }), 200
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run()
