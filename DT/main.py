import numpy as np
import sys
import pruning

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
        #return count, Leaf(labels[0]), depth
        return count, TreeNode(None, None, None, None, True, int(labels[0])), depth
    else:
        node = find_split(data)
        #if isinstance(node, TreeNode):
        if node.isLeaf == False:
            l_dataset = data[data[:, node.attribute] <= node.value]
            r_dataset = data[data[:, node.attribute] > node.value]
            count, node.left, l_depth = decision_tree_learning(count, l_dataset, depth+1)
            count, node.right, r_depth = decision_tree_learning(count, r_dataset, depth+1)

            return count, node, np.max([l_depth, r_depth])
        #elif isinstance(node, Leaf):
        elif node.isLeaf:
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
        decison_tree: learned function based

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


def evaluate(data):
    folds = create_folds(data)
    # split into test and training arrays
    accuracies = []
    for i in range(len(folds)):
        test = folds[i]
        training = np.concatenate(folds[:i] + folds[i+1:], axis = 0)
        count, tree, depth = decision_tree_learning(-1, training, 0)

        right = 0
        wrong = 0

        for i in range(test.shape[0]):
            guess = classify(test[i], tree)
            if test[i][-1] == guess:
                right += 1
            else:
                wrong += 1
        print(depth)

        accuracies.append( right / (right + wrong) )

    return accuracies


class TreeNode:
    def __init__(self, attribute, value, left, right, isLeaf = False, label = None):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.isLeaf = isLeaf
        self.label = label

    def isLeaf(self):
        return self.left==None and self.right==None

    def __str__(self):
        return ', '.join(['{key}={value}'.format(key=key, value=self.__dict__.get(key)) for key in self.__dict__])

class Leaf:
    def __init__(self, label):
        self.label = label

    def isLeaf(self):
        return True

    def __str__(self):
        return ', '.join(['{key}={value}'.format(key=key, value=self.__dict__.get(key)) for key in self.__dict__])

def main():
    # Load data: Arrays of 2000x8
    clean_data = np.loadtxt('wifi_db/clean_dataset.txt')
    noisy_data = np.loadtxt('wifi_db/noisy_dataset.txt')
    test_data = np.loadtxt('test_01.txt')
    depth_val = 0
    count = -1
    # print(mpla_data[:,-1])
    #count, x, depth_val = decision_tree_learning(count, clean_data, depth_val)
    #print(x)
    #Checking leaves
    #leafNodeCount = 0
    #leafNodeCount = pruning.findLeafNode(x)
    #print(leafNodeCount)
    # print(vars(find_split(test_data)))
    # print("Count:", count)
    print(evaluate(clean_data))


if __name__ == "__main__": main()
# print(find_split(clean_data).attribute)
