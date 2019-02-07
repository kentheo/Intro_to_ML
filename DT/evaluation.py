import numpy as np
from TreeNode import *

def create_folds(data, n_folds = 10):
    '''
    params:
        data: dataset to be split
        n_folds: number of folds to create, defaults to 10

    returns:
        list of folds, equal length if data is divisible by folds, else some
        folds will be longer by 1
    '''
    np.random.shuffle(data)
    folds = np.array_split(data, n_folds)
    return folds


def classify(sample, decision_tree):
    '''
    params:
        sample: input to classify
        decison_tree: TreeNode object to run over for classification

    returns:
        expected class label
    '''
    if decision_tree.isLeaf:
        return decision_tree.label
    else:
        feature, value = decision_tree.attribute, decision_tree.value
        if sample[feature] <= value:
            return classify(sample, decision_tree.left)
        else:
            return classify(sample, decision_tree.right)


def evaluate(data, tree):
    '''
    params:
        data: samples to run predictions on
        tree: learned function to use for prediction
        confusion: Boolean to determine whether to compute confusion matrix

    returns:
        overall prediction accuracy in range [0 1]
        confusion matrix (will be full of zeros if input confusion = False)
    '''
    right = 0
    wrong = 0
    confusion_matrix = np.zeros((4,4))
    for i in range(data.shape[0]):
        guess = classify(data[i], tree)
        if data[i][-1] == guess:
            right += 1
        else:
            wrong += 1

        confusion_matrix[int(data[i][-1]) - 1][int(guess) - 1] += 1
    accuracy = (right / (right + wrong))
    print("----------", confusion_matrix)
    return accuracy, confusion_matrix


def k_fold_cv(data, k = 10):
    folds = create_folds(data, k)
    # split into test and training arrays
    conf_matrices = []
    for i in range(len(folds)):
        test = folds[i]
        training = np.concatenate(folds[:i] + folds[i+1:], axis = 0)
        count, tree, depth = decision_tree_learning(-1, training, 0)
        acc, conf_matrix = evaluate(test, tree)
        conf_matrices.append(conf_matrix)
        print("Fold {}: Accuracy: {}".format(i, acc))
    return conf_matrices

# Returns all performance metrics
def evaluation(data, k = 10):
    conf_matrices = k_fold_cv(data, k)
    recall = np.zeros((k, 4))
    precision = np.zeros((k, 4))
    f1_score = np.zeros((k, 4))
    classification_rate = np.zeros(k)

    for i in range(len(conf_matrices)):
        matrix = conf_matrices[i]
        j = 0
        for j in range(4):
            recall[i][j] = compute_recall(matrix, j+1)
            precision[i][j] = compute_precision(matrix, j+1)
            f1_score[i][j] = compute_f1(recall[i][j], precision[i][j])
        classification_rate[i] = compute_classification_rate(matrix)

    avg_recall = np.mean(recall, axis=1)
    avg_precision = np.mean(precision, axis=1)
    avg_f1_score = np.mean(f1_score, axis=1)
    avg_class_rate = np.mean(classification_rate)

    return avg_recall, avg_precision, avg_f1_score, avg_class_rate

def compute_recall(confusion_matrix, label):
    '''
    :param confusion_matrix: Pre-computed confusion matrix for actual against predicted labels
    :param label: Class label to compute recall, should be in an int in range [1-4]
    :return: Recall value of class
    '''
    # Recall = TP / (TP + FN)
    print(confusion_matrix)
    i = label - 1
    tp = confusion_matrix[i][i]

    # Row of confusion matrix including TP, FN
    fn_tp = np.sum(confusion_matrix, axis=1)[i]
    print("fn_tp", fn_tp)
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
    fp_tp = np.sum(confusion_matrix, axis=0)[i]
    # print("fp_tp", fp_tp)
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
