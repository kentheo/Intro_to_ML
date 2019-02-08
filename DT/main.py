import numpy as np
import sys
import visualization
from TreeNode import *
from evaluation import *
from pruning import *

def main():
    # Default values for args
    task = 'creation'
    dataset = 'clean'
    # Otherwise use provided args
    if len(sys.argv[1:]) != 0:
        task = sys.argv[1:2][0].lower()
        dataset = sys.argv[2:3][0].lower()
    # Load data: Arrays of 2000x8
    if dataset == 'clean':
        data = np.loadtxt('wifi_db/clean_dataset.txt')
    elif dataset == 'noisy':
        data = np.loadtxt('wifi_db/noisy_dataset.txt')
    else:
        # !! Must be modified by tester !!
        data = np.loadtxt('to/be/provided/by/tester')

    if task == 'creation':
        # Initialisation of depth variable to be incremented by Decision Tree algorithm
        depth_val = 0
        # Method to produce a Tree
        tree, depth_val = decision_tree_learning(data, depth_val)
        print("Depth of created tree:", depth_val)
    elif task == 'visualization':
        # Initialisation of depth variable to be incremented by Decision Tree algorithm
        depth_val = 0
        # Method to produce a Tree
        tree, depth_val = decision_tree_learning(data, depth_val)
        # Visualization of Tree created above
        visualization.visualizeTree(x, depth_val)
    elif task == 'evaluation':
        avg_recall, avg_precision, avg_f1_score, avg_class_rate = evaluation(data)
        print('----------- Performance Metrics after K Fold Validation ------------------')
        print("Average Recall for each Class:", avg_recall)
        print("Average Precision for each Class:", avg_precision)
        print("Average F1 Score for each Class:", avg_f1_score)
        print("Average Classification Rate of K Fold Validation:", avg_class_rate)
    elif task == 'pruning':
        folds = create_folds(data)
        training = folds[0]
        testing = folds[2]
    	# Initialisation of depth variable to be incremented by Decision Tree algorithm
        depth_val = 0
        # Method to produce a Tree
        tree, depth_val = decision_tree_learning(training, depth_val)
        print("BEFORE PRUNING: Depth of created tree:", depth_val)

        pruned_tree = pruneTree(tree, folds[5])
        # print("AFTER PRUNING: Depth of created tree:", depth_val)

if __name__ == "__main__": main()
# print(find_split(clean_data).attribute)
