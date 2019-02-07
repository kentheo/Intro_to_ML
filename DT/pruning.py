from TreeNode import TreeNode
import numpy as np
from main import *
from evaluation import *

def findLeafNode(currentNode, head, data):
	'''
	params:
		currentNode: The current location in the decision tree
		head: The head of the decision tree
		data: The data used to evaluate the decision tree and determine if prune
		should occur or not

	returns:
		The count of the number of prunes that occurred during this iteration of tree
		evaluation.
	'''
	count = 0
	if not currentNode.left.isLeaf and not currentNode.right.isLeaf:
		count += findLeafNode(currentNode.left, head, data)
		count += findLeafNode(currentNode.right, head, data)
	else:
		print("Found node")

		acc1 = evaluate(data, head, False)
		print("ACC1: {}".format(acc1[0]))
		left = currentNode.left
		right = currentNode.right

		removeLeaves(currentNode)

		acc2 = evaluate(data, head, False)
		print("ACC2: {}".format(acc2[0]))
		if acc1[0] > acc2[0]:
			print("Shouldnt prune")
			currentNode.left = left
			currentNode.right = right
			currentNode.isLeaf = False
		else:
			print("Should prune")
			count += 1
	return count


def removeLeaves(parent):
	'''
	params:
		parent: The TreeNode which is having it's leaves removed
	'''
	parent.right = None
	parent.left = None
	parent.isLeaf = True
	return


if __name__ == "__main__": 
	clean_data = np.loadtxt('wifi_db/clean_dataset.txt')
	noisy_data = np.loadtxt('wifi_db/noisy_dataset.txt')
	
	folds = create_folds(clean_data)
	training = folds[0]
	testing = folds[2]

	np.append(training, folds[7])
	np.append(testing, folds[8])
	count, x, depth = decision_tree_learning(-1, training, 0)
	acc, cm = evaluate(clean_data, x, False)
	print(acc)
	nodes_pruned = 1
	while nodes_pruned > 0:
		nodes_pruned = findLeafNode(x, x, testing)
		acc, cm = evaluate(clean_data, x, False)
	
		print(acc)
		print(nodes_pruned)

