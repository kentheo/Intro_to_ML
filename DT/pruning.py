import numpy as np
from main import *
from TreeNode import TreeNode
from evaluation import *
from visualization import *

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
	depth = 0
	if not currentNode.left.isLeaf or not currentNode.right.isLeaf:
		if not currentNode.left.isLeaf:
			count += findLeafNode(currentNode.left, head, data)
		elif not currentNode.right.isLeaf:
			count += findLeafNode(currentNode.right, head, data)
	else:
		print("Found node")
		acc1, cm = evaluate(data, head)
		print("ACC1: {}".format(acc1))
		left = currentNode.left
		right = currentNode.right

		removeLeaves(currentNode)

		acc2, cm = evaluate(data, head)
		print("ACC2: {}".format(acc2))
		if acc1 > acc2:
			print("Shouldnt prune")
			currentNode.left = left
			currentNode.right = right
			currentNode.isLeaf = False
		else:
			print("Should prune")
			count += 1
	return count

def findDepth(tree):
	depth1 = 0
	depth2 = 0
	if not tree.left.isLeaf:
		#depth1 += 1
		depth1 += findDepth(tree.left)
	if not tree.right.isLeaf:
		#depth2 += 1
		depth2 += findDepth(tree.right)
	if depth1 > depth2:
		return depth1 + 1
	else:
		return depth2 + 1


def removeLeaves(parent):
	'''
	params:
		parent: The TreeNode which is having it's leaves removed
	'''
	parent.right = None
	parent.left = None
	parent.isLeaf = True
	return

def pruneTree(tree, validation_data):
	'''
		params:
			tree: The decision tree to be pruned
			validation_data: The dataset used to evaluate the tree performance
		returns:
			The pruned tree
	'''
	nodes_pruned = 1
	while nodes_pruned > 0:
		nodes_pruned = findLeafNode(tree, tree, validation_data)
		#print(depth)
		#Just here for testing purposes
		acc, cm = evaluate(validation_data, tree)
		#print(acc)
		#print("CM: {}".format(cm))
		print(nodes_pruned)

	return tree

if __name__ == "__main__":
	clean_data = np.loadtxt('wifi_db/clean_dataset.txt')
	noisy_data = np.loadtxt('wifi_db/noisy_dataset.txt')
	data = np.loadtxt('test_01.txt')
	folds = create_folds(clean_data)
	training = folds[0]
	testing = folds[2]
	#training = data[0:11]
	#testing = data[12:]

	
	np.append(training, folds[7])
	np.append(testing, folds[8])
	x, depth = decision_tree_learning(training, 0)
	#acc, cm = evaluate(clean_data, x)
	#print(acc)
	#print(findDepth(x))
	depth = findDepth(x)
	print("Depth: {}".format(depth))
	visualizeTree(x, depth)
	#tree = pruneTree(x, testing)
