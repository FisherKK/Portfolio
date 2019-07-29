# Dog Breed Prediction

The goal of this project is to provide a working prototype of a system which is capable of discriminating between dogs and humans. System shouldn't just choose between two classes dog and human but also return none class for each different image it classifies.

If image is labeled as dog or human, the system should provide next classification which is an assigment of the post similar dog-breed class to the given image.

There is no minimum requirement of how many types of dog breeds the system needs to support.

It includes following scripts:
- [start_server.py](start_server.py) - hosts flask webpage that allows to select image and receive classifications.

It includes following notebooks:
- [showcase.ipynb](showcase.ipynb) - data downloading, model building process and conclusions in one notebook

## Getting Started

### Prerequisites

Project requires Python 3.6.6 with installed [virtualenv](https://pypi.org/project/virtualenv/).

### Setup

1. Pull the project.
2. Setup new virtualenv with Python 3.6.6.
3. Install [requirements.txt](requirements.txt) file via pip.
4. Move to project directory.
5. Run jupyter notebook `Showcase.ipynb` to download data and generate models. As an alternative, already trained models 
can be downloaded from the following link https://www.dropbox.com/s/ryuqhmmmovjgs9f/TrainedModels.zip?dl=0 and copied 
into project.
6. Run script `start_server.py`.
7. Access Flask server webpage at `http://127.0.0.1:8887`.

## Model details


1. **Human Predictor** - Haar Cascade `haarcascade_frontalface_alt` included in OpenCV library. It is capable of 
detecting human faces. When human face is found on the images, then classifier flags that human can be also found on the 
image. It is quite naive solution but suprisingly gives `f1_score` of value **0.938** on testing dataset constructed 
from 500 human images and 500 dog images. Other facial cascades were tested too but selected one gave best results. 
It is also important to note that some dog images contain humans and this is the main weakness of this solution. 
Apart from that is quite straightforward and lightweight.


2. **Dog Predictor** - `ResNet50 which` is just loaded set of network weights from Keras library. Luckily `ResNet50` 
was trained on many anmial images. It is capable of detecting various dog breeds and because of that it can be used as 
dog detector. If for given dog image it returns a class corresponding to dog breed then it is possible to say that the 
dog is present on the image. Even if dog from unkown to ResNet50 breed is used for prediction it should still give dog 
breed class most similar to the given dog (as dogs are in general similar to each other). So in this case training of 
model was not needed. Model has great performance of `f1_score` value equal to **0.987**. Tested on dataset constructed 
from 500 human images and 500 dog images it managed to correctly detect 492 dogs and correclty ignore 496 people. 
Further investigation could be made to see on which human images network managed to fail. Maybe there was a human with 
a dog or some dog in background.


3. **Dog Breed Predictor** - finetuned neural network of `InceptionV3` architecture. InceptionV3 was loaded with 
"imagenet" weights through Keras library. Top with dense layers was thrown away and the rest was frozen. New top of 
neural network was constructed from `GlobalAveragePooling2D` layer and Dense layer with softmax output. It is to train 
neural network the new outputs which are dog breeds. Apart from `InceptionV3`, architectures like `VGG16` and `VGG19` 
were also tested. Best architecture was picked and the result is `f1_score` value of **0.785** on testing dataset - not 
available to network during training process.
---

#### Examples

<img src="https://github.com/FisherKK/Portfolio/blob/master/Projects/Dog-Breed-Prediction/images/example1.png" width="500px" height="auto"/>

---

<img src="https://github.com/FisherKK/Portfolio/blob/master/Projects/Dog-Breed-Prediction/images/example2.png" width="500px" height="auto"/>

---

<img src="https://github.com/FisherKK/Portfolio/blob/master/Projects/Dog-Breed-Prediction/images/example3.png" width="500px" height="auto"/>

---

<img src="https://github.com/FisherKK/Portfolio/blob/master/Projects/Dog-Breed-Prediction/images/example4.png" width="500px" height="auto"/>

### Flask webpage

The [start_server.py](start_server.py) launches interactive web application with file picker and possibility to get
model prediction.

---
<img src="https://github.com/FisherKK/Portfolio/blob/master/Projects/Dog-Breed-Prediction/images/application_example1.png" width="600px" height="auto"/>

---
<img src="https://github.com/FisherKK/Portfolio/blob/master/Projects/Dog-Breed-Prediction/images/application_example2.png" width="600px" height="auto"/>

---
<img src="https://github.com/FisherKK/Portfolio/blob/master/Projects/Dog-Breed-Prediction/images/application_example3.png" width="600px" height="auto"/>

## Built With

* [Udacity](https://www.udacity.com/) - project for passing Data Engineering section and Data Science Nanodegree

## License

This project is licensed under the MIT License.
