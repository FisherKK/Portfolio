import os

os.environ["KERAS_BACKEND"] = "theano"

from project.web.file_picker import save_picked_file
from project.model.Wrapper import HumanDogBreedPredictor

from project.config import (
    HUMAN_PREDICTOR_FILEPATH,
    DOGBREED_PREDICTOR_FILEPATH,
    DOG_BREED_CLASSES
)

from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
@app.route("/index")
def show_index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    global model
    filepath = save_picked_file()
    filename = filepath.split(os.path.sep)[-1]

    is_human, is_dog, dog_breed = model.predict(filepath, plot=False, verbose=False)

    if is_dog and is_human:
        message = "System got confused... it seems that it's a dog and human at the same time. Scary!"
    elif is_dog:
        message = "This is dog of breed: {}".format(dog_breed)
    elif is_human:
        message = "This is human that looks like dog of breed: {}".format(dog_breed)
    else:
        message = "This is neither dog or human."

    return render_template("complete.html", filename=filename, message=message)


def main():
    global model
    model = HumanDogBreedPredictor(HUMAN_PREDICTOR_FILEPATH, DOGBREED_PREDICTOR_FILEPATH, DOG_BREED_CLASSES)
    app.run(port=8887, debug=True)


if __name__ == "__main__":
    main()
