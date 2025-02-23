"""Metrics for classification.
"""

'''
Micro vs Macro average (for multi-class)

# Micro:
Each tp/fp/tn/fn are summed for all class.

# Macro:
average(all the class' own precision/recall value)

# Example:

Precision = tp / (tp + fp)

Class A: 1 TP and 1 FP
Class B: 10 TP and 90 FP
Class C: 1 TP and 1 FP
Class D: 1 TP and 1 FP

Precision_A = Precision_C = Precision_D = 0.5, Precision_B = 0.1
Macro_Ave_Precision = mean(precision of all class) = ((0.5 * 3) + 0.1)/4 = 0.4
Micro_Ave_Precision = sum(all tp)/sum(all tp + fp) = (1 + 10 + 1 + 1)/(2 + 100 + 2 + 2) = 0.123

Source: https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin

'''

import numpy as np

def get_confusion_matrix_val(y_true, y_pred, target):
    tp = np.sum(np.logical_and((y_pred == target), (y_true == y_pred)))
    tn = np.sum(np.logical_and((y_true != target), (y_true == y_pred)))

    fp = np.sum(np.logical_and((y_pred == target), (y_true != y_pred)))
    fn = np.sum(np.logical_and((y_true == target), (y_true != y_pred))) #
    return tp, tn, fp, fn

def make_onehot(y, labels):
    """Convert y into a one hot format

    For example, given:
        y = [1,2,3,2,2,3]
        labels = [1,2,3]
    It will return:
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1],
         [1, 0, 0],
         [0, 1, 0],
         [0, 1, 0],
         [0, 0, 1]]

    NOTE: You do NOT have to use this function. You MAY use it, if you find it
    helpful, especially for calculating precision and recall.

    Arguments
    ---------
        y: np.ndarray of shape (n_samples,)
        labels: list-like

    Returns
    -------
        np.ndarray: of shape (n_samples, len(labels))
    """

    labels = set(labels)
    if len(y.shape) != 1:
        raise Exception("Currently support only 1d input to make_onehot")

    label_indices = {label: i for i, label in enumerate(labels)}

    row_selector = [i for i, label in enumerate(y) if label in labels]
    column_selector = [label_indices[label] for label in y if label in label_indices]

    onehot = np.zeros((len(y), len(labels)), dtype=int)
    onehot[row_selector, column_selector] = 1
    return onehot


def check_metric_args(y_true, y_pred, average, labels):
    """Will check that y_true and y_pred are of compatible and correct shapes.
    
    Arguments
    ---------
        y_true: list-like 
        y_pred: list-like, of same shape as y_true
        average: One of "micro", "macro", or None
        labels: The labels for which we will calculate metrics

    Returns
    -------
        y_true: np.ndarray
        y_pred: np.ndarray
    """

    if average not in ["macro", "micro", None]:
        raise Exception("average param must be one of 'macro' or 'micro', or None.")

    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)

    if y_true.shape != y_pred.shape:
        raise Exception("shape of y_true and y_pred is not the same")

    return y_true, y_pred


def precision(y_true, y_pred, average, labels):
    """Calculate precision.

    `labels` will be used to 
    
    Arguments
    ---------
        y_true: list-like 
        y_pred: list-like, of same shape as y_true
        average: One of "micro", "macro", or None
        labels: The labels for which we will calculate metrics

    Returns
    -------
        np.ndarray or float:
            If `average` is None, it returns a numpy array of shape
            (len(labels), ), where the precision values for each class are 
            calculated.
            Otherwise, it returns either the macro, or micro precision value as 
            float.
    """
    y_true, y_pred = check_metric_args(y_true, y_pred, average, labels)

    ##### Write code here #######

    if average is None:
        precisions = []    
        for l in labels:
            tp, tn, fp, fn = get_confusion_matrix_val(y_true, y_pred, l)
            prec_v = tp / (tp + fp)
            precisions.append(prec_v)
        result = np.array(precisions)
    elif average == 'micro':
        sum_tp, sum_tn, sum_fp, sum_fn = 0, 0, 0, 0
        for l in labels:
            tp, tn, fp, fn = get_confusion_matrix_val(y_true, y_pred, l)
            sum_tp += tp
            sum_tn += tn
            sum_fp += fp
            sum_fn += fn
        result = sum_tp / (sum_tp + sum_fp)
    elif average == 'macro':
        precisions = []
        for l in labels:
            tp, tn, fp, fn = get_confusion_matrix_val(y_true, y_pred, l)
            if (tp + fp) == 0:
                prec_v = 0
            else:
                prec_v = tp / (tp + fp)
            precisions.append(prec_v)
        result = np.average(precisions)
    else:
        print("ERROR | recall | Unexpected average value ({})".format(average))

    ##### End of your work ######
    return result


def recall(y_true, y_pred, average, labels):
    """Calculate precision.

    `labels` will be used to 
    
    Arguments
    ---------
        y_true: list-like 
        y_pred: list-like, of same shape as y_true
        average: One of "micro", "macro", or None
        labels: The labels for which we will calculate metrics

    Returns
    -------
        np.ndarray or float:
            If `average` is None, it returns a numpy array of shape
            (len(labels), ), where the recall values for each class are 
            calculated.
            Otherwise, it returns either the macro, or micro recall value as 
            float.
    """

    y_true, y_pred = check_metric_args(y_true, y_pred, average, labels)

    ##### Write code here #######

    if average is None:
        recalls = []    
        for l in labels:
            tp, tn, fp, fn = get_confusion_matrix_val(y_true, y_pred, l)
            recall_v = tp / (tp + fn)
            recalls.append(recall_v)
        result = np.array(recalls)
    elif average == 'micro':
        sum_tp, sum_tn, sum_fp, sum_fn = 0, 0, 0, 0
        for l in labels:
            tp, tn, fp, fn = get_confusion_matrix_val(y_true, y_pred, l)
            sum_tp += tp
            sum_tn += tn
            sum_fp += fp
            sum_fn += fn
        result = sum_tp / (sum_tp + sum_fn)
    elif average == 'macro':
        recalls = []
        for l in labels:
            tp, tn, fp, fn = get_confusion_matrix_val(y_true, y_pred, l)
            recall_v = tp / (tp + fn)
            recalls.append(recall_v)
        result = np.average(recalls)
    else:
        print("ERROR | recall | Unexpected average value ({})".format(average))

    ##### End of your work ######

    return result


def test():
    """Test precision and recall
    """

    labels = [
        "blue",
        "red",
        "yellow",
    ]

    true1 = ["blue", "red", "blue", "blue", "blue", "blue", "yellow"]
    pred1 = ["blue", "red", "yellow", "yellow", "red", "red", "red"]

    # recall: tp/(tp + fn)

    # tp:
    # r: 1
    # b: 1
    # y: 0

    # fn:
    # r: 1, 0
    # b: 5, 5 (4)
    # y: 1, 1

    # r = 2/7 = 0.2857


    for (correct_precision, correct_recall, averaging) in [
        [0.4166666666666667, 0.39999999999999997, "macro"],
        [0.2857142857142857, 0.2857142857142857, "micro"],
    ]:
        our_recall = recall(true1, pred1, labels=labels, average=averaging)
        our_precision = precision(true1, pred1, labels=labels, average=averaging)

        print("\nAveraging: {}\n============".format(averaging))

        print("Recall\n-------")
        print("Correct: ", correct_recall)
        print("Ours: ", our_recall)
        print("")

        print("Precision\n---------")
        print("Correct: ", correct_precision)
        print("Ours: ", our_precision)
        print("")

        if correct_recall == our_recall and correct_precision == our_precision:
            print("All good!")
        else:
            print("Hmm, check implementation.")


if __name__ == "__main__":
    test()
