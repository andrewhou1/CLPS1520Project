import numpy as np
import tensorflow as tf
import math
import collections

categoryFile = open('ObjectClassesMap.txt', 'r')
RGBValues = list()
IDCategory = dict()
for line in categoryFile:
	nums = line.split()
	nums[0] = float(nums[0])
	nums[1] = float(nums[1])
	nums[2] = float(nums[2])
	nums[3] = int(nums[3])
	IDCategory[nums[3]] = nums[4]
	rowList = list()
	rowList.append(nums[0])
	rowList.append(nums[1])
	rowList.append(nums[2])
	RGBValues.append(rowList)


