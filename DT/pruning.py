from main import TreeNode
from main import Leaf
import numpy as np




def findLeafNode(currentNode, head):
	count = 0		
	if not currentNode.left.isLeaf() and not currentNode.right.isLeaf():
		count += findLeafNode(currentNode.left, head)
		count += findLeafNode(currentNode.right, head)
	else:
		print("Found node")
		#Perfrom prune test here
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

def removeLeaves(parent):
	parent.right = None
	parent.left = None
	return		

def calculateValidationError(head, trainingData, testData):
	#Calls evaluate() func to get error for each fold of 10-fold cross validation
	#averages to get "global error estimate" and returns this
	return 0

def calculateCVAccuracy(head):
	trainingDataset = np.loadtxt('wifi_db/clean_dataset.txt')
	folds = splitDataset(trainingDataset)
	global_error = 0
	for i in range(10):
		testFold = folds[i]
		#Gets other 9 folds for training/validation
		trainingFolds = folds - testFold

		global_error += calculateValidationError(head, trainingFolds, testFold)
	#Returns average global error
	return global_error/10


def splitDataset(dataset):
	fold_length = len(dataset) / 10
	folds = []
	#Splits dataset into 10 equal size folds
	for i in range(10):
		startIndex = int(i * fold_length)
		endIndex = int((i+1) * fold_length)
		folds.append(dataset[startIndex:endIndex])
	return folds

if __name__ == "__main__": calculateCVAccuracy(1)