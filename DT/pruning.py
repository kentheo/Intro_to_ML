from TreeNode import TreeNode
import numpy as np
from main import classify
import copy




def findLeafNode(currentNode, head, data):
	count = 0
	if not currentNode.left.isLeaf and not currentNode.right.isLeaf:
		count += findLeafNode(currentNode.left, head, data)
		count += findLeafNode(currentNode.right, head, data)
	else:
		print("Found node")
		#Perfrom prune test here
		pruneTest(head, currentNode, data)
		count += 1
	return count

#Works out if leaves should be pruned. Returns true if leaves pruned, false if tree is the same.

def pruneTest(head, node, data):
	local_tree = copy.deepcopy(head)
	pre_error = post_error = 0
	#pre_error = evaluate(data)
	left = node.left
	right = node.right
	#print(node)
	#print(local_tree == head)
	#removeLeaves(node)
	#print("POST")
	#print(local_tree == head)
	#print(node)
	#post_error = 0
	acc1 = compareTrees(local_tree, head, data)
	removeLeaves(node)
	acc2 = compareTrees(local_tree, head, data)
	print(acc1)
	print(acc2)
	if post_error < pre_error:
		print("Prune")
		#do the prune
	else:
		print("No Prune")
	return

def compareTrees(tree1, tree2, data):
	right1 = right2 = wrong1 = wrong2 = 0
	accuracy1 = 0
	accuracy2 = 0

	for i in range(len(data)):
		guess1 = classifyPrune(data[i], tree1)
		#print("Guess: {}".format(guess1))
		if data[i][-1] == guess1:
			right1 += 1
		else:
			wrong1 += 1

		# guess2 = classifyPrune(data[i], tree2)
		# if data[i][-1] == guess2:
		# 	right2 += 1
		# else:
		# 	wrong2 += 1
	accuracy1 = (right1 / (right1 + wrong1))

	# accuracy2 = (right2 / (right2 + wrong2))

	#print("Pre: {}".format(accuracy1))
	#print("Post: {}".format(accuracy2))
	return accuracy1

def classifyPrune(sample, decisionTree):
	
	if decisionTree.isLeaf:
		#print("Leaf")
		#print("Label: {}".format(decisionTree.label))
		return decisionTree.label
	else:
		#print("Not a leaf")
		feature, value = decisionTree.attribute, decisionTree.value
		#print("Feature: {}".format(feature))
		if sample[feature] <= value:
			return classifyPrune(sample, decisionTree.left)
		else:
			return classifyPrune(sample, decisionTree.right)


def removeLeaves(parent):
	parent.right = None
	parent.left = None
	parent.isLeaf = True
	parent.label = 2
	return

def calculateValidationError(head, trainingData, testData):
	#Calls evaluate() func to get error for each fold of 10-fold cross validation
	#averages to get "global error estimate" and returns this
	return 0


if __name__ == "__main__": calculateCVAccuracy(1)
