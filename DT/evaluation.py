import numpy as np


def compute_recall(confusion_matrix, label):
    '''
    :param confusion_matrix: Pre-computed confusion matrix for actual against predicted labels
    :param label: Class label to compute recall, should be in an int in range [1-4]
    :return: Recall value of class
    '''
    # Recall = TP / (TP + FN)
    i = label - 1
    tp = confusion_matrix[i][i]

    # Row of confusion matrix including TP, FN
    fn_tp = np.sum(confusion_matrix, axis=0)[i]

    return tp / fn_tp

def compute_precision(confusion_matrix, label):
    '''
    :param confusion_matrix: Pre-computed confusion matrix for actual against predicted labels
    :param label: Class label to compute precision
    :return: Precision value of class
    '''
    # Precision = TP / (TP + FP)
    i = label - 1
    tp = confusion_matrix[i][i]

    # Column of confusion matrix including TP, FP
    fp_tp = np.sum(confusion_matrix, axis=1)[i]

    return tp / fp_tp

def compute_f1(recall, precision):
    '''
    :param recall: Recall rate of a class
    :param precision: Precision rate of a class
    :return: F1 Measure
    '''
    return 2 * (precision * recall)/ (precision + recall)

def compute_classification_rate(confusion_matrix):
    '''
    :param confusion_matrix: Pre-computed confusion matrix for actual against predicted labels
    :return: Classification rate
    '''

    return np.trace(confusion_matrix) / np.sum(confusion_matrix)






