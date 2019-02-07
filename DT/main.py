import numpy as np
import sys
import pruning
import visualization
from TreeNode import *
from evaluation import *

def main():
    # Load data: Arrays of 2000x8
    clean_data = np.loadtxt('wifi_db/clean_dataset.txt')
    noisy_data = np.loadtxt('wifi_db/noisy_dataset.txt')
    test_data = np.loadtxt('test_01.txt')
    depth_val = 0
    count = -1
    # print(mpla_data[:,-1])
    # count, x, depth_val = decision_tree_learning(count, clean_data, depth_val)

    # visualization.visualizeTree(x, depth_val)

    print(evaluation(noisy_data))
    #print(x)
    #Checking leaves
    #leafNodeCount = 0
    #leafNodeCount = pruning.findLeafNode(x)
    #print(leafNodeCount)
    # print(vars(find_split(test_data)))
    # print("Count:", count)
    # print(evaluate(clean_data))
    #print(depth_val)
    # visualization.visualizeTree(visualization.createTestTree(5), 5)
    # samples = noisy_data[:200, :]
    # accuracy, confusion = run_samples(samples, x, True)
    # print(accuracy)
    # print(confusion)

    # samples_clean = clean_data[400:600, :]
    # accuracy_clean, conf_clean = run_samples(samples_clean, x, True)
    # print('Should be 100 %% since complete tree')
    # print(accuracy_clean)
    # print(conf_clean)


if __name__ == "__main__": main()
# print(find_split(clean_data).attribute)
