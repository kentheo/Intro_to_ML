import numpy as np
import sys

class TreeNode:
    def __init__(self, attribute, value, left, right, isLeaf = False, label = None):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.isLeaf = isLeaf
        self.label = label

    def __str__(self):
        return ', '.join(['{key}={value}'.format(key=key, value=self.__dict__.get(key)) for key in self.__dict__])

def find_number_labels(data):
    unique_labels = np.unique(data[:, -1])
    counter = []
    for i in unique_labels:
        counter.append(data[data[:, -1] == i].shape[0])
    return counter

def decision_tree_learning(data, depth):
    # Get all labels
    labels = data[:,-1]
    # Check if all samples have the same label
    result = len(set(labels)) == 1
    if result:
        # Return a leaf note with this value, depth
        return TreeNode(None, None, None, None, True, int(labels[0])), depth
    else:
        node = find_split(data)
        if node.isLeaf == False:

            # this should be adding label to tree as we go - NEED TO TEST
            node.label = plurality_vote(data)

            l_dataset = data[data[:, node.attribute] <= node.value]
            r_dataset = data[data[:, node.attribute] > node.value]
            node.left, l_depth = decision_tree_learning(l_dataset, depth+1)
            node.right, r_depth = decision_tree_learning(r_dataset, depth+1)

            return node, np.max([l_depth, r_depth])
        elif node.isLeaf:
            return node, depth
        else:
            print('Error!! Should not reach here!!')

def entropy(data):
    unique_labels = np.unique(data[:, -1])
    num_data = len(data)
    sum = 0
    for i in unique_labels:
        x = data[data[:, -1] == i].shape[0]
        p_k = x / num_data
        sum += p_k * np.log2(p_k)
    return -sum

def remainder(S_left, S_right):
    n_S_left = S_left.shape[0]
    n_S_right = S_right.shape[0]
    term1 = (n_S_left / (n_S_left + n_S_right)) * entropy(S_left)
    term2 = (n_S_right / (n_S_left + n_S_right)) * entropy(S_right)

    return term1 + term2

def information_gain(S_all, S_left, S_right):
    return entropy(S_all) - remainder(S_left, S_right)

def find_split(data):
    rows = data.shape[0]
    cols = data.shape[1]
    # Loop through sorted data and find feature vectors where examples in sorted
    # order that have different classifications
    max_gain = -sys.maxsize
    max_row, max_col = None, None

    for i in range(cols-1):
        indices = np.argsort(data[:, i])
        sorted_data = data[indices]

        labels = sorted_data[:, -1]
        indices = np.where(labels[:-1] != labels[1:])[0]
        for idx in indices:
            sval = sorted_data[idx, i]
            l_dataset = sorted_data[sorted_data[:, i] <= sval]
            r_dataset = sorted_data[sorted_data[:, i] > sval]
            if len(r_dataset) != 0:
                gain = information_gain(sorted_data, l_dataset, r_dataset)
                if gain > max_gain:
                    max_gain = gain
                    max_col = i
                    max_row = idx
                    split_value = sorted_data[idx, i]
                    # print("Max gain: {}, max_col: {}, max_row: {}, split_value: {}".format(max_gain, max_col, max_row, split_value))
    if max_gain > 0:
        return TreeNode(max_col, split_value, None, None)
    else:
        label = plurality_vote(data)
        #return Leaf(int(label))
        return TreeNode(None, None, None, None, True, int(label))

def plurality_vote(data):
    labels = data[:,-1]
    unique_labels = np.unique(labels)
    plurality = 0
    mc_label = -1
    for label in unique_labels:
        max_labels = len(labels[labels == label])
        if  max_labels > plurality:
            mc_label = label
            plurality = max_labels

    return mc_label
