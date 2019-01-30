from main import TreeNode
from main import Leaf




def findLeafNode(currentNode):
	count = 0		
	if not currentNode.left.isLeaf() and not currentNode.right.isLeaf():
		count += findLeafNode(currentNode.left)
		count += findLeafNode(currentNode.right)
	else:
		print("Found node")
		count += 1
	return count

def pruneTree():
	print("Gotta do some work")
	#navigate tree to leaf level -1 (node before leaves)
	#f

def calculateValidationError():
	return


if __name__ == "__main__": pruneTree()