'''

TODO:
 - Build own classifier
 - Confirm precision and recall methods are correct

'''

from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier

import our_metrics
import math

# Added lib
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

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


def load_data_file(data_file, is_test_file = False):
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
        if not is_test_file:
            if theme1_col in df.columns:
                y = df[theme1_col].tolist()

        print(
            "loaded {} lines {} labels ... done".format(
                len(X), "with" if y is not None else "without"
            )
        )

    return (X, y)

def load_balanced_file(data_file):
    with open(data_file) as f:
        df = pd.read_csv(f)
        df_x, df_y = df['x'].tolist(), df['y'].tolist()
        return df_x, df_y

def get_balance_class(X, y):
    df = pd.DataFrame({'x': X, 'y': y})
    ave = math.floor(np.mean(np.array(list(df.groupby('y').x.count()))))

    # ave = 120
    ave = 150
    # ave = 170   # Average of top 5.

    def sampling_k_elements(group, k=ave):
        if len(group) < k:
            return group
        return group.sample(k)

    df_bal = df.groupby('y').apply(sampling_k_elements).reset_index(drop=True)

    # print(df_bal['y'].value_counts())
    return df_bal['x'].to_list(), df_bal['y'].to_list()

def build_naive_bayes():
    """

    Returns
    -------
        Pipeline
        An sklearn Pipeline
    """
    nb_pipeline = None
    ##### Write code here #######

    nb_pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('clf', MultinomialNB(alpha=0.1)),
    ])

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

    logistic_pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('clf', LogisticRegression(multi_class='ovr')),
    ])

    ##### End of your work ######
    return logistic_pipeline


def build_random_forest():
    random_forest_pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('clf', RandomForestClassifier(n_estimators=10, max_depth=10, random_state=0)),
    ])
    return random_forest_pipeline

def build_linear_svc():
    linear_svc_pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer(use_idf=True)),
        ('clf', LinearSVC(multi_class='ovr')),
    ])
    return linear_svc_pipeline

def build_svc_2():
    sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer(use_idf=False)),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=0.0001, random_state=42, max_iter=5, tol=None)),
               ])
    return sgd

def build_own_pipeline() -> Pipeline:
    """

    Returns
    -------
        Pipeline
        An sklearn Pipeline
    """
    pipeline = None
    ##### Write code here #######

    pipeline = build_linear_svc()  

    ##### End of your work ######
    return pipeline


def output_predictions(result):
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

    np.savetxt("./predictions.tsv", np.array(result), fmt='%d')
    
    ##### End of your work ######

def testing(X_train, y_train_true, X_dev, y_dev_true):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.naive_bayes import MultinomialNB

    # Count Vectorizer
    train_vectorizer = CountVectorizer()
    X_cv_train = train_vectorizer.fit_transform(X_train)
    # print(vectorizer.get_feature_names())
    # print(X.toarray())

    # Tfid Transformer
    tf_train_transformer = TfidfTransformer(use_idf=False).fit(X_cv_train)
    X_train_tfidf = tf_train_transformer.transform(X_cv_train)
    # print(X_train_tf)

    # Fit model
    print("X_train_tfidf shape", X_train_tfidf.shape)

    # print(tfidf_transformer.idf_)
    # print(cv.get_feature_names())
    clf = MultinomialNB(alpha=0.1).fit(X_train_tfidf, y_train_true)

    # Predict validation set
    # dev_vectorizer = CountVectorizer()
    X_cv_dev = train_vectorizer.transform(X_dev)
    print("fname", len(train_vectorizer.get_feature_names()))

    # tf_dev_transformer = TfidfTransformer(use_idf=False).fit(X_cv_dev)
    X_dev_tfidf = tf_train_transformer.transform(X_cv_dev)
    print(tf_train_transformer.get_params())

    dev_pred = clf.predict(X_dev_tfidf)

    # np_pred = np.array(predicted)
    # np_y = np.array(y_dev_true)
    # print("Predicted:")
    # print(np_pred)
    # print("True:")
    # print(np_y)

    # print("Correct:", np.sum(np_pred == np_y))
    # print("Wrong", np.sum(np_pred != np_y))

    labels = list(np.unique(np.array(y_train_true)))

    print("Pipe = {}".format("Self"))
    for averaging in ['micro', 'macro']:
        our_recall = our_metrics.recall(y_dev_true, dev_pred, labels=labels, average=averaging)
        our_precision = our_metrics.precision(y_dev_true, dev_pred, labels=labels, average=averaging)
        print("\tAveraging = {}\n\t\tRecall = {}\n\t\tPrecision = {}".format(averaging, our_recall, our_precision))


def main(balance_class=False):
    # X_train_bal, y_train_bal_true = load_balanced_file(TRAIN_BALANCED_FILE)
    print("balance_class=", balance_class)

    X_train, y_train_true = load_data_file(TRAIN_FILE)
    #if balance_class:
    #    X_train, y_train_true = get_balance_class(X_train, y_train_true)

    X_dev, y_dev_true = load_data_file(DEV_FILE)
    X_test, _ = load_data_file(TEST_FILE, is_test_file=True)

    bayes_pipeline = build_naive_bayes()
    logistic_pipeline = build_logistic_regr()
    # random_forest_pipeline = build_random_forest()
    # linear_svc_pipeline = build_linear_svc()
    # svc_2_pipe = build_svc_2()
    own_pipeline = build_own_pipeline()

    for name, pipeline in (
        ["Naive Bayes", bayes_pipeline,],
        ["Logistic Regression", logistic_pipeline,],
        ["Own Pipeline (Linear SVC)", own_pipeline,],
    ):
        if pipeline is not None:

            ##### Write code here #######
            # Train model
            model = pipeline.fit(X_train, y_train_true)

            # Predict with Val set
            dev_pred = model.predict(X_dev)

            # Get Metrics
            labels = list(np.unique(np.array(y_train_true)))
            print("Pipe = {}".format(name))
            for averaging in ['micro', 'macro']:
                our_recall = our_metrics.recall(y_dev_true, dev_pred, labels=labels, average=averaging)
                our_precision = our_metrics.precision(y_dev_true, dev_pred, labels=labels, average=averaging)
                print("\tAveraging = {}\n\t\tRecall = {}\n\t\tPrecision = {}".format(averaging, our_recall, our_precision))

            if name == "Own Pipeline (Linear SVC)":
                # Run Test Set
                test_pred = model.predict(X_test)
                output_predictions(test_pred)

            ##### End of your work ######

    print("\n\n")
if __name__ == "__main__":
    main()
    # main(balance_class=True)
