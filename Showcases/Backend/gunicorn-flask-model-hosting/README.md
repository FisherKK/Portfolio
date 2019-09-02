# Gunicorn + Flask model hosting example

Project is capable of hosting, tf.keras and XGBClassifier models trained on MNIST dataset, and making them accessible
through the rest endpoint. 

Project contains three scrips:
- [train_models.py](train_models.py) - builds new tf.keras and XGBClassifier models and saves them in [model](model) 
directory in `.h5` and `.pkl` formats. 
- [start_flask_server.py](start_flask_server.py) - loads models and starts Flask server, exposing `/verify_vector_image`
endpoint that returns response from both models.
- [wsgi.py](wsgi.py) - wrapper for Flask app that will be used by Gunicorn.

## Local Server

### Prerequisites

Project requires Python 3.6.6 with installed [virtualenv](https://pypi.org/project/virtualenv/).

### Setup

1. Pull the project.
2. Setup new virtualenv with Python 3.6.6.
3. Install [requirements.txt](requirements.txt) file via pip.
4. Move to project directory.
5. Run [train_models.py](train_models.py) to spawn models in [model](model) directory.
6. Start Gunicorn server with command `gunicorn --bind <ip>:<port> --workers=<workers_num> wsgi:app`. 
7. Access endpoint server at `http://<ip>:<port>/verify_vector_image` endpoint.

### Gunicorn Launch Parameters
- for testing purposes, according to [config file](project/config/constants.py), `<ip>:<port>` was set to `0.0.0.0:8887`
- recommended `<workers_num>` is `(2*CPU) + 1` according to the [documentation](http://docs.gunicorn.org/en/stable/design.html?fbclid=IwAR3oB-YMwRJYdoBjLPc14pmaNd_BY2xkJZPHyrGPVEO3_l51MZGUR60kxSA#how-many-workers)
- at the same time documentation says that 4-12 should be enough

### Example Request

- Starting server `$ gunicorn --bind 0.0.0.0:8887 --workers=12  wsgi:app`
- Finding image for classification. In this case I've taken random image from MNIST dataset:
    
    <img src="https://github.com/FisherKK/Portfolio/blob/master/Showcases/Backend/gunicorn-flask-model-hosting/image/mnist_image_example.png" width="200" height="auto"/>
    
- Turning image into `.json` file.

```
{
  "image_vector": [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 18, 18,
    18, 126, 136, 175, 26, 166, 255, 247, 127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 36, 94, 154, 170, 253, 253, 253,
    253, 253, 225, 172, 253, 242, 195, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 49, 238, 253, 253, 253, 253, 253, 253, 253,
    253, 251, 93, 82, 82, 56, 39, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 219, 253, 253, 253, 253, 253, 198, 182, 247,
    241, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 156, 107, 253, 253, 205, 11, 0, 43, 154, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 1, 154, 253, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 139, 253, 190, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    11, 190, 253, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 241, 225, 160, 108,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 81, 240, 253, 253, 119, 25, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 186, 253, 253, 150, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 93, 252, 253, 187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 249, 253, 249, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 130, 183,
    253, 253, 207, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 39, 148, 229, 253, 253, 253, 250, 182, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 114, 221, 253, 253, 253, 253, 201, 78, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 66, 213, 253, 253, 253, 253, 198, 81, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 18, 171, 219, 253, 253, 253, 253, 195, 80, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 172, 226,
    253, 253, 253, 253, 244, 133, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 136, 253, 253, 253, 212,
    135, 132, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  ]
}
```

- Navigate to project root.
- Example curl request:
    - request command:
        ```
        curl -d "@testing/mnist_image_example.json" -H "Content-Type: application/json" -X POST http://0.0.0.0:8887/classify_image_vector
        ```
    
    - server response:
        ```
        {"mlp":{"predicted_number":5},"xgboost":{"predicted_number":5}}
        ```
  