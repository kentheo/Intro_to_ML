import numpy as np
import sys
# Load data: Arrays of 2000x8
clean_data = np.loadtxt('wifi_db/clean_dataset.txt')
noisy_data = np.loadtxt('wifi_db/noisy_dataset.txt')

first_mpla = clean_data[0:100,:]

def find_number_labels(data):
    unique_labels = np.unique(data[:, -1])
    counter = []
    for i in unique_labels:
        counter.append(data[data[:, -1] == i].shape[0])
    return counter
# print(first_mpla)
def decision_tree_learning(data, depth):
    # Get all labels
    labels = data[:,-1]

    # Check if all samples have the same label
    result = len(set(labels)) == 1

    if result:
        print("------------ All elements in list are same ------------------")
        # Return a leaf note with this value, depth
    else:
        print('mplaaaaaa')

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

    for i in range(cols-1):
        indices = np.argsort(data[:, i])
        sorted_data = data[indices]

        # Loop through sorted data and find feature vectors where examples in sorted
        # order that have different classifications
        max_gain = -sys.maxsize
        max_row, max_col = 0, 0
        labels = sorted_data[:, -1]
        indices = np.where(labels[:-1] != labels[1:])[0]

        for idx in indices:
            gain = information_gain(sorted_data, sorted_data[:idx+1, :], sorted_data[idx+1:, :])
            if gain > max_gain:
                max_gain = gain
                max_col = i
                max_row = idx
                split_value = sorted_data[idx, i]
        # for j in range(rows-1):
        #     current_label = sorted_data[j, -1]
        #     next_label = sorted_data[j+1, -1]
        #     # print(current_label)
        #     if current_label != next_label:
        #         gain = information_gain(sorted_data, sorted_data[:j+1, :], sorted_data[j+1:, :])
        #         if gain > max_gain:
        #             max_gain = gain
        #             max_col = i
        #             max_row = j
        #             split_value = sorted_data[j, i]
    # Split node
    return TreeNode(max_col, split_value, None, None)


#
# class DecisionTree:
#     def __init__(self):
#         self.depth =
#         self.

class TreeNode:
    def __init__(self, attribute, value, left, right):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right

print(find_split(clean_data).attribute)
