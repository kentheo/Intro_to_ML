import numpy as np
import sys
import visualization
from TreeNode import *
from evaluation import *
from pruning import *

def main():
    # For reproducability
    np.random.seed(1)
    # 3 Decimal points in all numpy print statements
    np.set_printoptions(precision=3)
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
    	# Whole path to unknown to us dataset text file
        data = np.loadtxt(sys.argv[2:3][0])

    print("Running {} on {} dataset...".format(task, dataset))
    # Check task to perform
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
        print("Depth of created tree:", depth_val)
        # Visualization of Tree created above
        visualization.visualizeTree(tree, depth_val)
    elif task == 'evaluation':
        avg_recall, avg_precision, avg_f1_score, avg_class_rate = evaluation(data)
        print('----------- Performance Metrics after K Fold Validation -----------')
        print("Average Recall for each Class:", avg_recall)
        print("Average Precision for each Class:", avg_precision)
        print("Average F1 Score for each Class:", avg_f1_score)
        print("Average Classification Rate of K Fold Validation: {:.3f}".format(avg_class_rate))
    elif task == 'pruning':
        avg_recall, avg_precision, avg_f1_score, avg_class_rate = evaluation(data, pruning = True)
        print('----------- Performance Metrics after K Fold Validation -----------')
        print("Average Recall for each Class:", avg_recall)
        print("Average Precision for each Class:", avg_precision)
        print("Average F1 Score for each Class:", avg_f1_score)
        print("Average Classification Rate of K Fold Validation: {:.3f}".format(avg_class_rate))

if __name__ == "__main__": main()
