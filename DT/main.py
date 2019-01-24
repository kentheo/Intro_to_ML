import numpy as np

# Load data: Arrays of 2000x8
clean_data = np.loadtxt('wifi_db/clean_dataset.txt')
noisy_data = np.loadtxt('wifi_db/noisy_dataset.txt')

first_mpla = clean_data[0:100,:]

def find_number_labels(data):
    unique_labels = np.unique(data[:, -1])
    counter = np.count(data[data[(len(data)-1) ]])
    return unique_labels
# print(first_mpla)
def decision_tree_learning(data, depth):
    # Get all labels
    labels = data[:,(data.shape[1]-1)]

    # Check if all samples have the same label
    result = len(set(labels)) == 1

    if result:
        print("------------ All elements in list are same ------------------")
        # Return a leaf note with this value, depth
    else:
        print('mplaaaaaa')

depth = 2
# decision_tree_learning(clean_data, depth)
find_number_labels(clean_data)

# def information_gain():
#     gain = entropy() - remainder()
#     return gain

# def entropy(attribute):
#     np.sum()
#     return
#
#
# class DecisionTree:
#     def __init__(self):
#         self.depth =
#         self.
#
#
# class TreeNode:
#     def __init__(self):
#         self.attribute =
#         self.value =
#         self.left =
#         self.right =
