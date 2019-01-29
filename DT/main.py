import numpy as np
import sys

def find_number_labels(data):
    unique_labels = np.unique(data[:, -1])
    counter = []
    for i in unique_labels:
        counter.append(data[data[:, -1] == i].shape[0])
    return counter

def decision_tree_learning(count, data, depth):
    count += 1
    # Get all labels
    labels = data[:,-1]
    # Check if all samples have the same label
    result = len(set(labels)) == 1
    if result:
        # Return a leaf note with this value, depth
        return count, Leaf(labels[0]), depth
    else:
        node = find_split(data)
        if isinstance(node, TreeNode):
            print(node.attribute)
            l_dataset = data[data[:, node.attribute] <= node.value]
            r_dataset = data[data[:, node.attribute] > node.value]
            count, node.left, l_depth = decision_tree_learning(count, l_dataset, depth+1)
            count, node.right, r_depth = decision_tree_learning(count, r_dataset, depth+1)

            return count, node, np.max([l_depth, r_depth])
        elif isinstance(node, Leaf):
            return count, node, depth
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
        return Leaf(int(label))

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



class TreeNode:
    def __init__(self, attribute, value, left, right):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right

    def isLeaf(self):
        return self.left==None and self.right==None

    def __str__(self):
        return ', '.join(['{key}={value}'.format(key=key, value=self.__dict__.get(key)) for key in self.__dict__])

class Leaf:
    def __init__(self, label):
        self.label = label

    def __str__(self):
        return ', '.join(['{key}={value}'.format(key=key, value=self.__dict__.get(key)) for key in self.__dict__])
def main():
    # Load data: Arrays of 2000x8
    clean_data = np.loadtxt('wifi_db/clean_dataset.txt')
    noisy_data = np.loadtxt('wifi_db/noisy_dataset.txt')
    test_data = np.loadtxt('test_01.txt')
    depth_val = 0
    count = -1
    mpla_data = clean_data[0:501, :]
    # print(mpla_data[:,-1])
    count, x, depth_val = decision_tree_learning(count, clean_data, depth_val)
    print(x)
    # print(vars(find_split(test_data)))
    # print("Count:", count)


if __name__ == "__main__": main()
# print(find_split(clean_data).attribute)
