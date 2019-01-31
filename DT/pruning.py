from main import TreeNode
from main import evaluate
import numpy as np

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
def pruneTree(head):
	local_tree = head
	pre_error = post_error = 0
	pre_error = calculateValidationError(head)

	#Remove leafNodes from local_tree
	post_error = calculateValidationError(local_tree)
	if pre_error > post_error:
		head = local_tree
		return True
	else:
		return False

def pruneTest(head, node, data):
	local_tree = head
	pre_error = post_error = 0
	pre_error = evaluate(data)
	left = node.left
	right = node.right
	removeLeaves(node)
	post_error = evaluate(data)
	print("pre: {}".format(pre_error))
	print("post: {}".format(post_error))

	return


def removeLeaves(parent):
	parent.right = None
	parent.left = None
	return		

def calculateValidationError(head, trainingData, testData):
	#Calls evaluate() func to get error for each fold of 10-fold cross validation
	#averages to get "global error estimate" and returns this
	return 0


if __name__ == "__main__": calculateCVAccuracy(1)