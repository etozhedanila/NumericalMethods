import math


def normC(matrix):
	result = []
	for row in matrix:
		sum = 0
		if type(row) != list:
			result.append(abs(row))
			continue
		for elem in row:
			sum += abs(elem)
		result.append(sum)
	return max(result)

def norm1(matrix):
	result = []
	if type(matrix[0]) != list:
		result = 0
		for elem in matrix:
			result += abs(elem)
		return result
	for j in range(len(matrix[0])):
		sum = 0
		for i in range(len(matrix)):
			sum += abs(matrix[i][j])
		result.append(sum)
	return max(result)

def norm2(matrix):
	sum = 0
	for row in matrix:
		if type(row) != list:
			sum += row*row
			continue
		for elem in row:
			sum += elem*elem
	return math.sqrt(sum)

def subtractVectors(v1,v2):
	result = []
	if len(v1) == len(v2):
		for i in range(len(v1)):
			result.append(v1[i] - v2[i])
	return result

def addVectors(v1,v2):
	result = []
	if len(v1) == len(v2):
		for i in range(len(v1)):
			result.append(v1[i] + v2[i])
	return result

def multiplyMatrix(m1,m2):
	result = []
	
	if type(m1[0]) == list and type(m2[0]) != list:
		
		for row in m1:
			sum = 0
			for i in range(len(row)):
				sum += row[i] * m2[i]
			result.append(sum)
		return result


	if type(m1[0]) != list and type(m2[0]) != list:
		for i in range(len(m1)):
			tmp = []
			for j in range(len(m2)):
				tmp.append(m1[i] * m2[j])
			result.append(tmp)
		return result


	if len(m1[0]) != len(m2):
		print("матрицы не могут быть перемножены")
		return

	
	for i in range(len(m1)):
		tmp = []
		for j in range(len(m2[0])):
			sum = 0
			for k in range(len(m1[0])):
				sum += m1[i][k] * m2[k][j]
			tmp.append(round(sum,5))
		result.append(tmp)
	return result

def transpose(matrix):
	result = []
	for i in range(len(matrix[0])):
		tmp = []
		for j in range(len(matrix)):
			tmp.append(matrix[j][i])
		result.append(tmp)
	return result