import numpy as np
import sys

class TreeNode:
    def __init__(self, attribute, value, left, right, isLeaf = False, label = None):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.isLeaf = isLeaf
        self.label = label

    def __str__(self):
        return ', '.join(['{key}={value}'.format(key=key, value=self.__dict__.get(key)) for key in self.__dict__])


# class TreeNode:
# 	def __init__(self, attribute, value, left, right):
# 	    self.attribute = attribute
# 	    self.value = value
# 	    self.left = left
# 	    self.right = right
#
# 	def addLeftChild(self, left):
# 		self.left = left
#
# 	def addRightChild(self, right):
# 		self.right = right
#
# 	def isLeaf(self):
# 		return self.left == None and self.right == None
