import pandas as pd

from project.config import (
    DEFAULT_MESSAGES_FILE_PATH,
    DEFAULT_CATEGORIES_FILE_PATH,
    MESSAGES_LINK,
    CATEGORIES_LINK

)

if __name__ == "__main__":
    df_messages = pd.read_csv(MESSAGES_LINK)
    df_messages.to_csv(DEFAULT_MESSAGES_FILE_PATH, index=False)
    print("Saved messages file to: {}".format(DEFAULT_MESSAGES_FILE_PATH))

    df_categories = pd.read_csv(CATEGORIES_LINK)
    df_categories.to_csv(DEFAULT_CATEGORIES_FILE_PATH, index=False)
    print("Saved categories file to: {}".format(DEFAULT_CATEGORIES_FILE_PATH))

