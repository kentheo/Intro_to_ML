from TreeNode import TreeNode
import numpy as np
import main
import evaluation




def findLeafNode(currentNode, head, data):
	count = 0
	if not currentNode.left.isLeaf and not currentNode.right.isLeaf:
		count += findLeafNode(currentNode.left, head, data)
		count += findLeafNode(currentNode.right, head, data)
	else:
		print("Found node")
		#Perfrom prune test here

		acc1 = evaluation.evaluate(data, head, True)
		print("ACC1: {}".format(acc1[0]))
		left = currentNode.left
		right = currentNode.right

		removeLeaves(currentNode)

		acc2 = evaluation.evaluate(data, head, True)
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
	parent.right = None
	parent.left = None
	parent.isLeaf = True
	return


if __name__ == "__main__": 
	clean_data = np.loadtxt('wifi_db/clean_dataset.txt')
	noisy_data = np.loadtxt('wifi_db/noisy_dataset.txt')
	
	folds = main.create_folds(clean_data)
	training = folds[0]
	testing = folds[2]

	np.append(training, folds[7])
	np.append(testing, folds[8])
	count, x, depth = main.decision_tree_learning(-1, training, 0)
	acc, cm = evaluation.evaluate(clean_data, x, True)
	print(acc)
	nodes_pruned = 1
	while nodes_pruned > 0:
		nodes_pruned = findLeafNode(x, x, testing)
		acc, cm = evaluation.evaluate(clean_data, x, True)
	
		print(acc)
		print(nodes_pruned)

