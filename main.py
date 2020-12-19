import tensorflow as tf
from flask import Flask, jsonify, render_template, flash, request, redirect

from source.ModelGeneral import ModelGeneral

print("Tensorflow version : " + tf.__version__)

# Set CPU as available physical device
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices=my_devices, device_type='CPU')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
my_model = None

if my_model is None:
    my_model = ModelGeneral()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_file(request_):
    # check if the post request has the file part
    if 'file' not in request_.files:
        flash('No file part')
        return redirect(request_.url)
    file = request_.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request_.url)
    if file and allowed_file(file.filename):
        return file
    return None


@app.route('/', methods=['GET', 'POST'])
def home_page():
    if request.method == 'POST':
        file = get_file(request_=request)
        if file is not None:
            try:
                labels = my_model.get_lables()
                xception = my_model.prediction(model="Xception",
                                               image_request=file)
                inceptionv3 = my_model.prediction(model="InceptionV3",
                                                  image_request=file)
                densenet201 = my_model.prediction(model="DenseNet201",
                                                  image_request=file)
            except():
                return jsonify({
                    "success": False,
                    "message": "File not exist!"
                }), 404

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


@app.route('/densenet201', methods=['POST'])
def result_densenet201():
    file = get_file(request_=request)
    if file is not None:
        try:
            labels = my_model.get_lables()
            densenet201 = my_model.prediction(model="DenseNet201",
                                              image_request=file)
        except():
            return jsonify({
                "success": False,
                "message": "File not exist!"
            }), 404

        return jsonify({
            "success": True,
            "message": "Predicted Results",
            "Labels": labels.tolist(),
            "DenseNet201": [round(num, 2) for num in densenet201[0].tolist()],
            "DenseNet201 Predicted": densenet201[1]
        }), 200
    else:
        return jsonify({
            "success": False,
            "message": "File not exist!"
        }), 404


@app.route('/inceptionv3', methods=['POST'])
def result_inceptionv3():
    file = get_file(request_=request)
    if file is not None:
        try:
            labels = my_model.get_lables()
            inceptionv3 = my_model.prediction(model="InceptionV3",
                                              image_request=file)
        except():
            return jsonify({
                "success": False,
                "message": "File not exist!"
            }), 404

        return jsonify({
            "success": True,
            "message": "Predicted Results",
            "Labels": labels.tolist(),
            "InceptionV3": [round(num, 2) for num in inceptionv3[0].tolist()],
            "InceptionV3 Predicted": inceptionv3[1]
        }), 200
    else:
        return jsonify({
            "success": False,
            "message": "File not exist!"
        }), 404


@app.route('/xception', methods=['POST'])
def result_xception():
    file = get_file(request_=request)
    if file is not None:
        try:
            labels = my_model.get_lables()
            xception = my_model.prediction(model="Xception",
                                           image_request=file)
        except():
            return jsonify({
                "success": False,
                "message": "File not exist!"
            }), 404

        return jsonify({
            "success": True,
            "message": "Predicted Results",
            "Labels": labels.tolist(),
            "Xception": [round(num, 2) for num in xception[0].tolist()],
            "Xception Predicted": xception[1]
        }), 200
    else:
        return jsonify({
            "success": False,
            "message": "File not exist!"
        }), 404


if __name__ == '__main__':
    app.run()
    if my_model is None:
        my_model = ModelGeneral()
