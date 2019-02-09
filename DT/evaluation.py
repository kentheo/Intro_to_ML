import numpy as np
import copy
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
    return accuracy, confusion_matrix


def k_fold_cv(data, k = 10, pruning = False):
    folds = create_folds(data, k)
    conf_matrices = []
    # print(folds)
    if pruning:
        for i in range(len(folds)):
            # Get a test set
            train_val_sets = copy.deepcopy(folds)
            test = folds[i]
            # Remove the test set from the copy of the folds
            del train_val_sets[i]
            for j in range(len(train_val_sets)):
                # Split training and validation sets
                validation = train_val_sets[j]
                training = np.concatenate(train_val_sets[:j] + train_val_sets[j+1:], axis = 0)
                tree, depth = decision_tree_learning(training, 0)
                # Prune
                pruned_tree = pruneTree(tree, validation)
                # After pruning, evaluate the pruned tree on the test set
                acc, conf_matrix = evaluate(test, pruned_tree)
                conf_matrices.append(conf_matrix)
                print("Fold {}, Validation {}: Accuracy: {}".format(i, j, acc))
    else:
        for i in range(len(folds)):
            test = folds[i]
            training = np.concatenate(folds[:i] + folds[i+1:], axis = 0)
            tree, depth = decision_tree_learning(training, 0)
            acc, conf_matrix = evaluate(test, tree)
            conf_matrices.append(conf_matrix)
            print("Fold {}: Accuracy: {}".format(i, acc))

    return conf_matrices

# Perform Step 3: Evaluation as a whole
def evaluation_step3(data, k = 10):
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

    avg_recall = np.mean(recall, axis=0)
    avg_precision = np.mean(precision, axis=0)
    avg_f1_score = np.mean(f1_score, axis=0)
    avg_class_rate = np.mean(classification_rate)

    return avg_recall, avg_precision, avg_f1_score, avg_class_rate

# Currently, main function uses this method
def evaluation(data, k = 10, pruning = False):
    conf_matrices = k_fold_cv(data, k, pruning)
    avg_conf_matrix = np.zeros((4, 4))
    print("{} Confusion matrices".format(len(conf_matrices)))
    for i in range(len(conf_matrices)):
        avg_conf_matrix += conf_matrices[i]
    avg_conf_matrix /= len(conf_matrices)

    # Compute avg performance metrics for each class
    avg_recall = np.zeros(4)
    avg_precision = np.zeros(4)
    avg_f1_score = np.zeros(4)
    avg_class_rate = 0.0
    for i in range(4):
        avg_recall[i] = compute_recall(avg_conf_matrix, i+1)
        avg_precision[i] = compute_precision(avg_conf_matrix, i+1)
        avg_f1_score[i] = compute_f1(avg_recall[i], avg_precision[i])

    avg_class_rate = compute_classification_rate(avg_conf_matrix)

    return avg_recall, avg_precision, avg_f1_score, avg_class_rate

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
    fn_tp = np.sum(confusion_matrix, axis=1)[i]
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

from pruning import *
