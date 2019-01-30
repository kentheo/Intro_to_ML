import matplotlib.pyplot as plt
from matplotlib import collections  as mc
import random
# Set up fig, ax

class TreeNode:
	def __init__(self, attribute, value, left, right):
	    self.attribute = attribute
	    self.value = value
	    self.left = left
	    self.right = right

	def addLeftChild(self, left):
		self.left = left

	def addRightChild(self, right):
		self.right = right

	def isLeaf(self):
		return self.left == None and self.right == None

def visualizeTree(head, depth):

	if head == None:
		return

	xmin = 0
	xmax = (2 ** depth) / 10
	ymin = 0
	ymax = 0.1 * depth

	fig, ax = plt.subplots(figsize=(2 ** depth, depth))
	ax.set_xlim([xmin,xmax])
	ax.set_ylim([ymin,ymax])

	x = xmax/2
	y = ymax - 0.05

	drawTree(head, depth - 1, x, y, fig, ax, None)

	plt.show()


def drawTree(head, depth, x, y, fig, ax, LR):
	# Prepare Parameters
	lines = []
	x_offset = (2 ** depth) / 20
	y_offset = 0.08

	if not head.isLeaf():

		string = "A" + str(head.attribute) + " < " + str("{:.1f}".format(head.value))

		if head.left != None:
			p1 = [x, y - 0.01]
			p2 = [x - x_offset, y - y_offset + 0.02]
			lines.append([p1, p2])

			drawTree(head.left, depth - 1, x - x_offset, y - y_offset, fig, ax, 'L')
		if head.right != None:
			p1 = [x, y - 0.01]
			p2 = [x + x_offset, y - y_offset + 0.02]
			lines.append([p1, p2])

			drawTree(head.right, depth - 1, x + x_offset, y - y_offset, fig, ax, 'R')

		lc = mc.LineCollection(lines, colors='red', linewidths=1, zorder=5)
		ax.add_collection(lc)

	else:
		string = "leaf: "
		if LR == 'L':
			string += "1.00"
		elif LR == 'R':
			string += "0.00"
	
	ax.text(x, y, string, color='black', horizontalalignment='center', zorder=10,
	        bbox=dict(facecolor='none', edgecolor='blue'))

def createTestTree(depth):
	
	ranAt1 = int(random.random() * 10)
	ranVal1 = random.random() * 100
	head = TreeNode(ranAt1, ranVal1, None, None)

	if depth > 1:
		head.addLeftChild(createTestTree(depth - 1))
		head.addRightChild(createTestTree(depth - 1))

	return head

if __name__ == '__main__':
	# Make a test tree

	depth = 4
	head = createTestTree(depth)

	# Draw Figure
	visualizeTree(head, depth=depth)
