import cv2
import numpy as np

from source.ModelDenseNet201 import ModelDenseNet201
from source.ModelInceptionV3 import ModelInceptionV3
from source.ModelXception import ModelXception


class ModelGeneral:
    image_size = 224
    labels = None
    floatx = r'float64'
    labelspath = r'static/label/Labels.npy'
    model_path = r'static/model_cnn/models/'
    weights_path = r'static/model_cnn/weights/'

    def __init__(self):
        self.__load_labels()
        if True:
            self.__load_model()
        else:
            self.__load_weight()

    @classmethod
    def __load_labels(cls):
        cls.labels = np.load(cls.labelspath)
        print("Load labels successfully!")

    @classmethod
    def get_lables(cls):
        return cls.labels

    @classmethod
    def __load_weight(cls):
        # Load model Xception
        ModelXception.set_path(model_path=None,
                               weight_path=cls.weights_path + r'WeightsXception.h5')
        cls.model_xception = ModelXception.get_model()
        # Load model InceptionV3
        ModelInceptionV3.set_path(model_path=None,
                                  weight_path=cls.weights_path + r'WeightsInceptionV3.h5')
        cls.model_inceptionv3 = ModelInceptionV3.get_model()
        # Load model DenseNet201
        ModelDenseNet201.set_path(model_path=None,
                                  weight_path=cls.weights_path + r'WeightsDensenet201.h5')
        cls.model_densenet201 = ModelDenseNet201.get_model()
        print("Load models successfully!")

    @classmethod
    def __load_model(cls):
        # Load model Xception
        ModelXception.set_path(model_path=cls.model_path + r'ModelXception.h5',
                               weight_path=None)
        cls.model_xception = ModelXception.get_model()
        # Load model InceptionV3
        ModelInceptionV3.set_path(model_path=cls.model_path + r'ModelInceptionV3.h5',
                                  weight_path=None)
        cls.model_inceptionv3 = ModelInceptionV3.get_model()
        # Load model DenseNet201
        ModelDenseNet201.set_path(model_path=cls.model_path + r'ModelDensenet201.h5',
                                  weight_path=None)
        cls.model_densenet201 = ModelDenseNet201.get_model()
        print("Load models successfully!")

    @classmethod
    def __resize_image(cls, image_path, image_size=224):
        img_arr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img_arr = cv2.resize(img_arr, (image_size, image_size))  # Reshaping images to preferred size
        img_arr = np.array(img_arr, dtype=cls.floatx) / 255
        img_arr = img_arr.reshape(-1, image_size, image_size, 3)
        return img_arr

    @classmethod
    def __prediction_classify(cls, model, path):
        img_arr = cls.__resize_image(path, cls.image_size)
        predictions = model.predict(img_arr)
        classes = np.argmax(predictions, axis=1)
        return [np.round(predictions[0] * 100, 2), cls.labels[classes[0]]]

    @classmethod
    def prediction(cls, model="Xception", image_path=None):
        if model == "Xception":
            return cls.__prediction_classify(model=ModelXception.get_model(),
                                             path=image_path)
        elif model == "InceptionV3":
            return cls.__prediction_classify(model=ModelInceptionV3.get_model(),
                                             path=image_path)
        elif model == "DenseNet201":
            return cls.__prediction_classify(model=ModelDenseNet201.get_model(),
                                             path=image_path)
        else:
            return None
