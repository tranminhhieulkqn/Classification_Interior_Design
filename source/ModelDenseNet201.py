import threading

from keras.applications import DenseNet201
from keras.layers import Dense, MaxPool2D, Flatten
from keras.models import Sequential, load_model


class ModelDenseNet201:
    __instance_model = None  # Instance model
    model_path = None  # Model path model
    weight_path = None  # Weight path model
    labels = 5
    image_size = 224

    def __init__(self):
        if (ModelDenseNet201.__instance_model is None) \
                and ((ModelDenseNet201.model_path is not None) or (ModelDenseNet201.weight_path is not None)):
            if ModelDenseNet201.model_path is not None:
                ModelDenseNet201.__instance_model = load_model(self.model_path)
            else:
                model = ModelDenseNet201.__create_model()
                model.load_weights(ModelDenseNet201.weight_path)
                ModelDenseNet201.__instance_model = model
        else:
            ModelDenseNet201.__instance_model = self

    @staticmethod
    def set_path(model_path=None, weight_path=None, labels=5, image_size=224):
        ModelDenseNet201.labels = labels
        ModelDenseNet201.image_size = image_size
        ModelDenseNet201.model_path = model_path
        ModelDenseNet201.weight_path = weight_path
        if model_path is None and weight_path is None:
            print(r'The path is not correct!')

    @classmethod
    def __create_model(cls):
        pre_trained_model = DenseNet201(input_shape=(cls.image_size, cls.image_size, 3),
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
        if ModelDenseNet201.__instance_model is None:
            with threading.Lock():
                if ModelDenseNet201.__instance_model is None:
                    ModelDenseNet201()
        return ModelDenseNet201.__instance_model
