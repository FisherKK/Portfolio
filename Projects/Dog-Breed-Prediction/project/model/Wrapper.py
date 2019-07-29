import cv2

import matplotlib.pyplot as plt
import numpy as np

from keras.preprocessing import image
from keras.models import load_model
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
from keras.applications.inception_v3 import preprocess_input as inceptionv3_preprocess_input


class HumanDogBreedPredictor:
    """Class which combines 'haarcascade_frontalface_alt', 'ResNet50' and finetuned
    'InceptionV3' models into system that detects humans and dogs and returns dog breed
    for the image."""

    def __init__(self, human_predictor_filepath, dog_breed_predictor_filepath, dog_names):
        """Class constructor.

        Parameters:
        -----------
        human_predictor_filepath: str
            Path to .xml file of 'haarcascade_frontalface_alt' cascade.
        dog_breed_predictor_filepath: str
            Path to .h5 keras model file.
        dog_names: list
            List containing dog breed names in the same order as softmax output of
            trained dog breed precition network.

        Returns:
        -----------
        None
        """
        self.human_predictor = self._init_human_predictor(human_predictor_filepath)
        print("Succesfully loaded human predictor!")

        self.dog_predictor = self._init_dog_predictor()
        print("Succesfully loaded dog predictor!")

        self.dogbreed_predictor = self._init_dog_breed_predictor(dog_breed_predictor_filepath)
        print("Succesfully loaded dog_breed predictor!")

        self.dog_names = dog_names

    def _init_human_predictor(self, path):
        """Method which loads human predictor.

        Parameters:
        -----------
        path: str
            Path to .xml file of 'haarcascade_frontalface_alt' cascade.

        Returns:
        -----------
        human_predictor: cv2.CascadeClassifier
            Loaded cascade file wrapped in OpenCV class.
        """
        human_predictor = cv2.CascadeClassifier(path)
        return human_predictor

    def _init_dog_predictor(self):
        """Method which loads dog predictor.

        Parameters:
        -----------
        None

        Returns:
        -----------
        dog_predictor: Sequential
            Keras ResNet50 model.
        """
        dog_predictor = ResNet50(weights="imagenet")
        return dog_predictor

    def _init_dog_breed_predictor(self, path):
        """Method which loads dog breed predictor.

        Parameters:
        -----------
        path: str
            Path to .h5 keras model file.

        Returns:
        -----------
        dog_breed_predictor: Sequential
            Finetuned InceptionV3 Keras model.
        """
        dog_breed_predictor = load_model(path, compile=False)
        return dog_breed_predictor

    @staticmethod
    def _path_to_tensor(img_path, img_size=(224, 224)):
        """Method takes image from specific filepath, resizes it and saves as numpy.ndarray.

        Parameters:
        -----------
        img_path: str
            Filepath to image file.
        img_size: tuple
            Tuple to which loaded image will be resized.

        Returns:
        -----------
        img: numpy.ndarray
            Returns loaded and resized image.
        """
        img = image.load_img(img_path, target_size=img_size)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        return img

    def _human_prediction(self, img_path):
        """For given image path returns flag whether it contains human or not.

        Parameters:
        -----------
        img_path: str
            Filepath to image file.

        Returns:
        -----------
        result: int
            Value 1 if image contains human, 0 otherwise.
        """
        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = int(len(self.human_predictor.detectMultiScale(img_gray)) > 0)
        return result

    def _dog_prediction(self, img_path):
        """For given image path returns flag whether it contains dog or not.

        Parameters:
        -----------
        img_path: str
            Filepath to image file.

        Returns:
        -----------
        result: int
            Value 1 if image contains human, 0 otherwise.
        """
        img = self._path_to_tensor(img_path)
        img = resnet50_preprocess_input(img)
        result = np.argmax(self.dog_predictor.predict(img))
        result = int((result <= 268) and (result >= 151))
        return result

    def _dogbreed_prediction(self, img_path):
        """For given image path returns id of dog breed class.

        Parameters:
        -----------
        img_path: str
            Filepath to image file.

        Returns:
        -----------
        result: int
            Id of dog breed class.
        """
        img = self._path_to_tensor(img_path)
        img = inceptionv3_preprocess_input(img)
        result = np.argmax(self.dogbreed_predictor.predict(img))
        return result

    def predict(self, img_path, plot=False, verbose=False):
        """Method which for given image path loads image, makes prediction
        with each prediction and based on the results returns communicate to user.

        Parameters:
        -----------
        img_path: str
            Filepath to image file.
        plot: bool
            If set to true, sent image will be also displayed.
        verbose: bool
            If set to true, function will display messages to user.

        Returns:
        -----------
        is_human: int
            Information whether image contains human.
        is_dog: int
            Information whether image contains dog.
        dog_breed: str
            Name of the dog breed.
        """
        print("Making prediction for: {}".format(img_path))

        is_human = self._human_prediction(img_path)
        is_dog = self._dog_prediction(img_path)
        dog_breed = self.dog_names[self._dogbreed_prediction(img_path)]

        if plot:
            img = cv2.imread(img_path)
            cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(cv_rgb)
            plt.show()

        if verbose:
            if is_dog and is_human:
                print("\t- System got confused... it seems that it's a "
                      + "dog and human at the same time. Scary!")
            elif is_dog:
                print("\t- This is dog of breed: {}".format(dog_breed))
            elif is_human:
                print("\t- This is human that looks like dog of breed: {}".format(dog_breed))
            else:
                print("\t- This is neither dog or human.")

        return is_human, is_dog, dog_breed
