import os
from project.config import MODEL_DIR


def create_dir(dir_):
    temp = ""
    for part in dir_.split(os.sep):
        temp = os.path.join(temp, part)
        if not os.path.exists(temp):
            os.mkdir(temp)


if not os.path.exists(MODEL_DIR):
    create_dir(MODEL_DIR)
