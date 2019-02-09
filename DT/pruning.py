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
		acc1, cm = evaluate(data, head)
		left = currentNode.left
		right = currentNode.right

		removeLeaves(currentNode)

		acc2, cm = evaluate(data, head)
		if acc1 > acc2:
			currentNode.left = left
			currentNode.right = right
			currentNode.isLeaf = False
		else:
			count += 1
	return count

def findDepth(tree):
	depth1 = 0
	depth2 = 0
	if not tree.left.isLeaf:
		depth1 += findDepth(tree.left)
	if not tree.right.isLeaf:
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

	return tree

# Returns the number of nodes in a tree
def countNodes(root):

	count = 0

	if root is not None:
		count = 1

		if root.left is not None:
		    count += countNodes(root.left)
		if root.right is not None:
		    count += countNodes(root.right)
	return count

# if __name__ == "__main__":
#
# 	clean_data = np.loadtxt('wifi_db/clean_dataset.txt')
# 	noisy_data = np.loadtxt('wifi_db/noisy_dataset.txt')
# 	data = np.loadtxt('test_01.txt')
# 	folds = create_folds(noisy_data)
# 	print("folds: {}".format(len(folds)))
#
# 	training = np.concatenate(folds[1:10], axis = 0)
# 	testing = folds[0]
#
# 	print("train: {}".format(training.shape))
# 	print("test: {}".format(testing.shape))
#
# 	x, depth = decision_tree_learning(training, 0)
# 	tree1_str = str(x)
# 	nodesX = countNodes(x)
# 	depth_est = findDepth(x)
#
# 	print("Depth Returned: {}, Depth Estimated: {}".format(depth, depth_est))
# 	# visualizeTree(x, depth)
# 	tree = pruneTree(x, testing)
# 	depth2 = findDepth(tree)
#
# 	print("Depth Returned Pre-Prune: {}".format(depth))
# 	print("No of Nodes Pre-Pruning", nodesX)
# 	print("Pruned Depth: {}".format(depth2))
# 	print("No of Nodes Post-Pruning", countNodes(tree))
# 	# visualizeTree(tree, depth2)
# 	tree2_str = str(tree)
# 	print("Equals x-tree:", tree1_str == tree2_str)
