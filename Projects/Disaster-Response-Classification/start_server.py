import json
import argparse

from flask import (
    Flask,
    render_template,
    request
)

import plotly
from plotly.graph_objs import Bar

from sklearn.externals import joblib

from project.preprocessing.feature_engineering import tokenize

from project.utils.db_utils import (
    get_database_engine,
    sql_table_to_df
)

from project.config import (
    DEFAULT_DB_OUTPUT_FILE_PATH,
    DEFAULT_MODEL_OUTPUT_FILE_PATH,
    SQL_TABLE_NAME
)

parser = argparse.ArgumentParser()
parser.add_argument("--db_path",
                    help="Absolute path for .db file from where data should be loaded.",
                    default=DEFAULT_DB_OUTPUT_FILE_PATH,
                    type=str)

parser.add_argument("--model_path",
                    help="Absolute path from where model should be loaded.",
                    default=DEFAULT_MODEL_OUTPUT_FILE_PATH,
                    type=str)

app = Flask(__name__)


def add_category_count_graph(df):
    category_counts = df[[c for c in df.columns if "category" in c]].sum(axis=0)
    category_names = [c[9:] for c in category_counts.index]
    category_counts = category_counts.values

    return {
        "data": [
            Bar(
                x=category_names,
                y=category_counts
            )
        ],

        "layout": {
            "title": "Number of messages per category",
            "yaxis": {
                "title": "Count"
            },
            'height': 600,
            'margin': dict(b=200, pad=4),
        }
    }


def add_source_count_graph(df):
    genre_counts = df["genre"].value_counts()
    genre_names = genre_counts.index
    genre_counts = genre_counts.values

    return {
        "data": [
            Bar(
                x=genre_names,
                y=genre_counts
            )
        ],

        "layout": {
            "title": "Source of gathered messages",
            "yaxis": {
                "title": "Count"
            },
            'height': 600,
            'margin': dict(b=200, pad=4),
        }
    }


@app.route("/")
@app.route("/index")
def index():
    graphs = [ add_source_count_graph(df_data), add_category_count_graph(df_data)]

    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('master.html', ids=ids, graphJSON=graphJSON)


@app.route("/predict")
def predict():
    message = request.args.get("query", "")

    classification_labels = model.predict([message])[0]

    labels = [c[9:] for c in df_data.columns if "category" in c]
    classification_results = dict(zip(labels, classification_labels))

    return render_template("predict.html", query=message, classification_result=classification_results)


def main():
    global df_data, model

    args = parser.parse_args()

    df_data = sql_table_to_df(SQL_TABLE_NAME, get_database_engine(args.db_path))
    model = joblib.load(DEFAULT_MODEL_OUTPUT_FILE_PATH)

    app.run(host="0.0.0.0", port=3001, debug=True)


if __name__ == "__main__":
    main()
