import threading

from keras.applications import InceptionV3
from keras.layers import Dense, MaxPool2D, Flatten
from keras.models import Sequential, load_model


class ModelInceptionV3:
    __instance_model = None  # Instance model
    model_path = None  # Model path model
    weight_path = None  # Weight path model
    labels = 5
    image_size = 224

    def __init__(self):
        if (ModelInceptionV3.__instance_model is None) and (ModelInceptionV3.model_path is not None):
            if ModelInceptionV3.model_path is not None:
                ModelInceptionV3.__instance_model = load_model(self.model_path)
            else:
                model = ModelInceptionV3.__create_model()
                model.load_weights(ModelInceptionV3.weight_path)
                ModelInceptionV3.__instance_model = model
        else:
            ModelInceptionV3.__instance_model = self

    @staticmethod
    def set_path(model_path=None, weight_path=None, labels=5, image_size=224):
        ModelInceptionV3.labels = labels
        ModelInceptionV3.image_size = image_size
        ModelInceptionV3.model_path = model_path
        ModelInceptionV3.weight_path = weight_path
        if model_path is None and weight_path is None:
            print(r'The path is not correct!')

    @classmethod
    def __create_model(cls):
        pre_trained_model = InceptionV3(input_shape=(cls.image_size, cls.image_size, 3),
                                        include_top=False,
                                        weights="imagenet")
        model = Sequential([
            pre_trained_model,
            MaxPool2D((2, 2), strides=2),
            Flatten(),
            Dense(cls.labels, activation='softmax')])
        model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    @staticmethod
    def get_model():
        if ModelInceptionV3.__instance_model is None:
            with threading.Lock():
                if ModelInceptionV3.__instance_model is None:
                    ModelInceptionV3()
        return ModelInceptionV3.__instance_model
