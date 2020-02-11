from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression

import our_metrics

TRAIN_FILE = Path("raw_data/GunViolence/train.tsv")
DEV_FILE = Path("raw_data/GunViolence/dev.tsv")
TEST_FILE = Path("raw_data/GunViolence/test.tsv")

LABELS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# These frames/labels correspond to
# 1) Gun or 2nd Amendment rights
# 2) Gun control/regulation
# 3) Politics
# 4) Mental health
# 5) School or public space safety
# 6) Race/ethnicity
# 7) Public opinion
# 8) Society/culture
# 9) Economic consequences


def load_data_file(data_file):
    """Load newsframing data

    Returns
    -------
    tuple
        First element is a list of strings(headlines)
        If `data_file` has labels, the second element
        will be a list of labels for each headline. 
        Otherwise, the second element will be None.
    """
    print("Loading from {} ...".format(data_file.name), end="")
    text_col = "news_title"
    theme1_col = "Q3 Theme1"

    with open(data_file) as f:
        df = pd.read_csv(f, sep="\t")
        X = df[text_col].tolist()

        y = None
        if theme1_col in df.columns:
            y = df[theme1_col].tolist()

        print(
            "loaded {} lines {} labels ... done".format(
                len(X), "with" if y is not None else "without"
            )
        )
    return (X, y)


def build_naive_bayes():
    """

    Returns
    -------
        Pipeline
        An sklearn Pipeline
    """
    nb_pipeline = None
    ##### Write code here #######

    ##### End of your work ######
    return nb_pipeline


def build_logistic_regr():
    """

    Returns
    -------
        Pipeline
        An sklearn Pipeline
    """
    logistic_pipeline = None
    ##### Write code here #######

    ##### End of your work ######
    return logistic_pipeline


def build_own_pipeline() -> Pipeline:
    """

    Returns
    -------
        Pipeline
        An sklearn Pipeline
    """
    pipeline = None
    ##### Write code here #######

    ##### End of your work ######
    return pipeline


def output_predictions(pipeline):
    """Load test data, predict using given pipeline, and write predictions to file.

    The output must be named "predictions.tsv" and must have the following format.
    Here, the first three examples were predicted to be 7,2,3, and the last were 
    predicted to be 6,6, and 2.

    Be sure not to permute the order of the examples in the test file.

        7
        2
        3
        .
        .
        .
        6
        6
        2

    """
    ##### Write code here #######
    pass
    ##### End of your work ######


def main():
    X_train, y_train_true = load_data_file(TRAIN_FILE)
    X_dev, y_dev_true = load_data_file(DEV_FILE)

    bayes_pipeline = build_naive_bayes()
    logistic_pipeline = build_logistic_regr()

    for name, pipeline in (
        ["Naive Bayes", bayes_pipeline,],
        ["Logistic Regression", logistic_pipeline,],
    ):
        if pipeline is not None:

            ##### Write code here #######
            continue
            ##### End of your work ######


if __name__ == "__main__":
    main()
