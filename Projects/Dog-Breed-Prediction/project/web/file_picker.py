import os
from flask import request

from project.config import DATA_DIR


def save_picked_file():
    """Retrieves file from file-picker form and saves in DATA_DIR.

    Parameters:
    ----------
    None

    Returns:
    ----------
    filepath: str
        Filepath to saved file.
    """
    file = request.files.getlist("file")[0]
    filepath = os.path.join(DATA_DIR, file.filename)
    file.save(filepath)

    return filepath
