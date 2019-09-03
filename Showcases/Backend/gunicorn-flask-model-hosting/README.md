# Gunicorn + Flask model hosting example

Project is capable of hosting, tf.keras and XGBClassifier models trained on MNIST dataset, and making them accessible
through the rest endpoint. 

Scripts overview:
- [train_models.py](train_models.py) - builds new tf.keras and XGBClassifier models and saves them in [model](model) 
directory in `.h5` and `.pkl` formats. 
- [start_flask_server.py](start_flask_server.py) - loads models and starts Flask server, exposing `/verify_vector_image`
endpoint that returns response from both models.
- [inference_speed_test.py](inference_speed_test.py) - loads models and performs 1000 predictions for the same image, 
returns averaged prediction time
- [wsgi.py](wsgi.py) - wrapper for Flask app that will be used by Gunicorn.

## Models

Models are already included in project [model](model) directory, as the files are small.

### Training models

Run [train_models.py](train_models.py) to spawn new models in [model](model) directory.

### Model Performance

Performance on MNIST dataset:
- tf.keras
    ```
    Model results:
	 train | loss: 0.01267085460151577, accuracy: 0.9959166646003723
	   val | loss: 0.13343262002021947, accuracy: 0.9762499928474426
	  test | loss: 0.11085368104698634, accuracy: 0.9781000018119812
    ```
- XGBClassifier
    ```
    Model results:
	 train | accuracy: 0.990125
	  test | accuracy: 0.964
    ```
    
### Inference Times
According to [inference_speed_test.py](inference_speed_test.py) script:
- mlp is capable of making ~833 predictions on single thread:
    ```
    1000 inference trials took on average '0.0012' (min: 0.0011, max: 0.0351, std: 0.0011) seconds.
    ```
- xgboost is capable of making ~1429 predictions on single thread:
    ```
    1000 inference trials took on average '0.0007' (min: 0.0005, max: 0.0378, std: 0.0012) seconds.
    ```

## Endpoint Info

Model response is available at `http://0.0.0.0:8887/verify_vector_image` url after server is started.

## Request Info

Request of `POST` type with ContentType equal to `application/json`. 

Sent JSON should have following format:

```
{
  "image_vector": [...]
}
```

Filled example of such vector can be found [here](testing/mnist_image_example.json), it also can be used for testing whether service works or not.

## Gunicorn Launch Parameters
- recommended `<workers_num>` is `(2 * CPU) + 1` according to the [documentation](http://docs.gunicorn.org/en/stable/design.html?fbclid=IwAR3oB-YMwRJYdoBjLPc14pmaNd_BY2xkJZPHyrGPVEO3_l51MZGUR60kxSA#how-many-workers)
- at the same time documentation says that 4-12 should be enough

## Local Server

### Prerequisites

Project requires Python 3.6.6 with installed [virtualenv](https://pypi.org/project/virtualenv/).

### Setup

1. Pull the project.
2. Setup new virtualenv with Python 3.6.7.
3. Install [requirements.txt](requirements.txt) file via pip.
4. Move to project directory.
5. Start Gunicorn server with command `gunicorn --bind 0.0.0.0:8887 --workers=<workers_num> wsgi:app`. 
6. Access endpoint server at `http://0.0.0.0:8887/verify_vector_image` endpoint.
        
## Local Server with Docker

### Prerequisites

Requires [Docker](https://www.docker.com/) app installed and started.

### Setup

1. Pull the project.
2. Setup new virtualenv with Python 3.6.7.
3. Install [requirements.txt](requirements.txt) file via pip.
4. Move to project directory.
5. Edit Gunicorn parameters inside [Dockerfile](Dockerfile) CMD line. Default number of workers is set to 12.
6. Build docker container with the following command `$ docker build --tag flask_gunicorn_app .`.
7. Start docker process by running `$ docker run --detach -p 8887:8887 flask_gunicorn_app`.
8. Access endpoint server at `http://0.0.0.0:8887/verify_vector_image` endpoint.

## Example Request

### Data Preparation

- Finding image for classification. In this case I've taken random image from MNIST dataset:
    
    <img src="https://github.com/FisherKK/Portfolio/blob/master/Showcases/Backend/gunicorn-flask-model-hosting/image/mnist_image_example.png" width="200" height="auto"/>

- Turning image into `.json` file.

```
{
  "image_vector": [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 18, 18, 18, 126, 136, 175, 26, 166, 
    255, 247, 127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 36, 94, 154, 170, 253, 253, 253, 253, 253, 
    225, 172, 253, 242, 195, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 49, 238, 253, 253, 253, 253, 253, 
    253, 253, 253, 251, 93, 82, 82, 56, 39, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 219, 253, 253, 253, 
    253, 253, 198, 182, 247, 241, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 156, 107, 
    253, 253, 205, 11, 0, 43, 154, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 1, 154, 
    253, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 139, 253, 190, 
    2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 190, 253, 70, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 241, 225, 160, 108, 1, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 81, 240, 253, 253, 119, 25, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 186, 253, 253, 150, 27, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 93, 252, 253, 187, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 249, 253, 249, 64, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 130, 183, 253, 253, 207, 2, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 39, 148, 229, 253, 253, 253, 250, 182, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 114, 221, 253, 253, 253, 253, 201, 78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 23, 66, 213, 253, 253, 253, 253, 198, 81, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 18, 171, 219, 253, 253, 253, 253, 195, 80, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 55, 172, 226, 253, 253, 253, 253, 244, 133, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 136, 253, 253, 253, 212, 135, 132, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  ]
```
- Example file is available here [mnist_image_example.json](testing/mnist_image_example.json).

### Local Run

- Starting server `$ gunicorn --bind 0.0.0.0:8887 --workers=12  wsgi:app`.
  
### Docker Run

- Starting docker.
- Starting server `$ docker run --detach -p 8887:8887 flask_gunicorn_app`.
- Checking if server is running on proper port by calling `$ docker ps`.

```
MacBook-Pro-Kamil:~ kamilkrzyk$ docker ps
CONTAINER ID        IMAGE                COMMAND                  CREATED             STATUS              PORTS                    NAMES
96888ebcc280        flask_gunicorn_app   "gunicorn --bind 0.0…"   4 seconds ago       Up 2 seconds        0.0.0.0:8887->8887/tcp   eager_nightingale
```

### Getting Response

- Navigate to project root.
- Example curl request:
    - request command:
        ```
        curl -d "@testing/mnist_image_example.json" -H "Content-Type: application/json" -X POST http://0.0.0.0:8887/classify_image_vector
        ```
    
    - server response:
        ```
        {"mlp":{"predicted_number":5},"xgboost":{"predicted_number":5}}