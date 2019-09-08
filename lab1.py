import copy
import math
from matrix import *

#1_1
def LU():
	print("введите размерность")
	n = int(input()) 
	print("введите матрицу")
	a = [[float(j) for j in input().split()] for i in range(n)]
	u = [0] * n
	m = [[0] * n for i in range(n)]
	for k in range(n-1):	#k - номер шага
		for i in range(k+1, n):		#i - номер строки
			u[i] = a[i][k] / a[k][k]
			m[i][k] = u[i]
			for j in range(k, n):		#j - номер столбца
				a[i][j] = a[i][j] - u[i]*a[k][j]
	print()
	
	for i in range(n):
		m[i][i] = 1
		for j in range(i+1,n):
			m[i][j] = 0
		

	print("матрица L")
	for r in m:
		print(' '.join([str(elem) for elem in r]))

	print("матрица U")
	for r in a:
		print(' '.join([str(elem) for elem in r]))

	return m, a

def solutionWithLU():

	l, u = LU()
	n = len(l)
	z = []
	print("введите вектор правых частей")
	b = [float(s) for s in input().split()]
	z.append(b[0])

	for i in range(1, n):
		sum = 0
		for j in range(i):
			sum += z[j] * l[i][j]
		z.append(b[i] - sum)

	x = [0] * n

	x[n-1] = z[n-1]/u[n-1][n-1]
	
	for i in range(n-2, -1, -1):
		sum = 0
		for j in range(i+1, n):
			sum += u[i][j]*x[j]
		x[i] = (z[i] - sum) / u[i][i]

	print(x)
	




	
#1_2
def methodOfRun():
	print("введите вектор правых частей")
	d = [float(s) for s in input().split()]
	p = []
	q = []
	print("введите ненулевые коэфициенты матрицы системы")
	for i in range(len(d)):
		row = [float(s) for s in input().split()]
		if i == 0:
			tmpP = -row[1] / row[0]
			tmpQ = d[i] / row[0]
			p.append(tmpP)
			q.append(tmpQ)
			continue
		if i == len(d) - 1:
			p.append(0)
			tmpQ = (d[i] - row[0]*q[i-1]) / (row[1] + row[0]*p[i-1])
			q.append(tmpQ)
			continue
		tmpQ = (d[i] - row[0]*q[i-1]) / (row[1] + row[0]*p[i-1])
		tmpP = -row[2] / (row[1] + row[0]*p[i-1])
		q.append(tmpQ)
		p.append(tmpP)

	print("P = ", p)
	print("Q = ", q)

	x = [0]*len(d)
	x[-1] = q[-1]
	for i in range(len(d) - 2, -1, -1):
		x[i]=(p[i]*x[i+1]+q[i])
	print("X = ", x)








def methodOfSimpleIterations():
	print("введите точность вычислений")
	accuracy = float(input())
	print("введите вектор правых частей")
	d = [float(s) for s in input().split()]
	b = []
	a = []
	print("введите матрицу системы")
	inputMatrix = []
	for i in range(len(d)):
		row = [float(s) for s in input().split()]
		inputMatrix.append(row)
		b.append(d[i] / inputMatrix[i][i])
		tmp = []
		for j in range(len(row)):
			if i != j:
				tmp.append(-inputMatrix[i][j] / inputMatrix[i][i])
			if i == j:
				tmp.append(0)
		a.append(tmp)
	norm = None 
	flag = True
	if normC(a) > 1:
		if norm1(a) > 1:
			if norm2(a) > 1:
				print("достаточное условие не выполнено")
				flag = False
				#return
			norm = norm2
		else:
			norm = norm1
	else:
		norm = normC
	normA = norm(a)
	if flag == True:
		print("достаточное условие выполнено, ||a|| = ", normA)
	x = b.copy()
	k = 0
	while True:
		k += 1
		xPrev = x.copy()
		x = addVectors(b, multiplyMatrix(a,xPrev))
		print("x[", k, "] = ", x)
		if flag == True:
			tmpAccuracy = (normA / (1 - normA)) * norm(subtractVectors(x,xPrev))
		else:
			tmpAccuracy = norm(subtractVectors(x,xPrev))
		print("e[", k, "] = ",tmpAccuracy)
		if tmpAccuracy < accuracy:
			print("процесс сошелся за ", k, "итераций")
			break

def methodOfZeidel():
	print("введите точность вычислений")
	accuracy = float(input())
	print("введите вектор правых частей")
	d = [float(s) for s in input().split()]
	b = []
	a = []
	print("введите матрицу системы")
	inputMatrix = []
	for i in range(len(d)):
		row = [float(s) for s in input().split()]
		inputMatrix.append(row)
		b.append(d[i] / inputMatrix[i][i])
		tmp = []
		for j in range(len(row)):
			if i != j:
				tmp.append(-inputMatrix[i][j] / inputMatrix[i][i])
			if i == j:
				tmp.append(0)
		a.append(tmp)
	norm = None 
	
	if normC(a) > 1:
		if norm1(a) > 1:
			if norm2(a) > 1:
				print("достаточное условие не выполнено")
				return
			norm = norm2
		else:
			norm = norm1
	else:
		norm = normC
	normA = norm(a)

	print("достаточное условие выполнено, ||a|| = ", normA)
	x = b.copy()
	k = 0
	
	c = copy.deepcopy(a)
	for i in range(len(c)):
		for j in range(len(c[i])):
			if i > j:
				c[i][j] = 0
	normOfC = norm(c)
	
	while True:
		k += 1
		xPrev = x.copy()
		for i in range(len(x)):
			x[i] = b[i] 
			for j in range(i):
				x[i] += a[i][j] * x[j]
				
			for j in range(i+1,len(a[i])):
				x[i] += a[i][j] * xPrev[j]
				
		print("x[", k, "] = ", x)
		tmpAccuracy = (normOfC / (1 - normA)) * norm(subtractVectors(x,xPrev))
		print("e[", k, "] = ",tmpAccuracy)
		if tmpAccuracy < accuracy:
			print("процесс сошелся за ", k, "итераций")
			break

#1_4

# def transpose(matrix):
# 	result = []
# 	for i in range(len(matrix[0])):
# 		tmp = []
# 		for j in range(len(matrix)):
# 			tmp.append(matrix[j][i])
# 		result.append(tmp)
# 	return result

def t(matrix):
	sum = 0
	for i in range(len(matrix)):
		for j in range(i + 1,len(matrix)):
			sum += matrix[i][j] * matrix[i][j]
	return math.sqrt(sum)




def methodOfRotation():
	print("введите точность вычислений")
	accuracy = float(input())
	print("введите размерность")
	n = int(input()) 
	print("введите матрицу")
	a = [[float(j) for j in input().split()] for i in range(n)]
	
	u = []
	k = 0
	tmpA = copy.deepcopy(a)
	while t(tmpA) > accuracy:

		k+=1
		maxElement = tmpA[0][1]
		maxI, maxJ = 0, 1
		for i in range(n-1):
			for j in range(i + 1,len(tmpA[i])):
				if abs(tmpA[i][j]) > maxElement:
					maxElement = abs(tmpA[i][j])
					maxI, maxJ = i, j

		if tmpA[maxI][maxI] != tmpA[maxJ][maxJ]:
			fi = 0.5 * math.atan(2*tmpA[maxI][maxJ] / (tmpA[maxI][maxI] - tmpA[maxJ][maxJ]))
		else:
			fi = math.pi / 4

		tmpU = []
		for i in range(n):
			tmp = []
			for j in range(n):
				if i == j and (i == maxI or j == maxJ):
					tmp.append(math.cos(fi))
					continue
				if i == j:
					tmp.append(1)
					continue
				if i == maxI and j == maxJ:
					tmp.append(-math.sin(fi))
					continue
				if i == maxJ and j == maxI:
					tmp.append(math.sin(fi))
					continue
				tmp.append(0)
			tmpU.append(tmp)


		tmpA = multiplyMatrix(multiplyMatrix(transpose(tmpU), tmpA), tmpU)
		if len(u) > 0:
			u = multiplyMatrix(u, tmpU)
		else:
			u = copy.deepcopy(tmpU)

		print("A = ")
		for row in tmpA:
			print(row)
	print("U = ")
	for row in u:
		print(row)

#1_5
def sign(x):
	if x < 0:
		return -1
	if x > 0:
		return 1
	if x == 0:
		return 0

def QRdecomposition(m):
	Q = []
	matrix = copy.deepcopy(m)
	n = len(matrix)
	for i in range(n - 1):

		tmpH = []

		tmpV = [0] * n

		for j in range(i):
			tmpV[j] = 0

		sum = 0
		for j in range(i, n):
			sum += matrix[j][i] * matrix[j][i]

		tmpV[i] = matrix[i][i] + sign(matrix[i][i]) * math.sqrt(sum)

		for j in range(i + 1, n):
			tmpV[j] = matrix[j][i] 

		c = 0
		for elem in tmpV:
			c += elem*elem
		c = 2 / c

		newV = multiplyMatrix(tmpV, tmpV)
		
		for i in range(n):
			tmp = []
			for j in range(n):
				if i == j:
					tmp.append(1 - c * newV[i][j])
				else:
					tmp.append(-c * newV[i][j])
			tmpH.append(tmp)


		matrix = multiplyMatrix(tmpH, matrix)

		if len(Q) > 0:
			Q = multiplyMatrix(Q, tmpH)
		else:
			Q = copy.deepcopy(tmpH)

	
	return Q, matrix

def checkTriangleQR(matrix, e):
	sum = 0
	for i in range(1, len(matrix)):
		for j in range(i):
			sum += matrix[i][j]*matrix[i][j]
	return math.sqrt(sum) > e

def checkDifference(m1,m2,e):
	if len(m1) == len(m2):
		for i in range(len(m1)):
			if abs(m1[i][i] - m2[i][i]) <= e:
				return False
	return True

def methodQR():
	print("введите точность вычислений")
	accuracy = float(input())
	print("введите размерность")
	n = int(input()) 
	print("введите матрицу")
	a = [[float(j) for j in input().split()] for i in range(n)]
	Q, R = QRdecomposition(a)
	oldA = []
	while checkDifference(oldA,a,accuracy) and checkTriangleQR(a, accuracy):
		
		oldA = copy.deepcopy(a)
		a = multiplyMatrix(R, Q)
		Q, R = QRdecomposition(a)
		
	for row in a:
		print(row)
		
	





def menu():
	print("1 LU-разложение ----- 1")
	print("2 Метод прогонки ---- 2")
	print("3.1 Метод простых итераций ---- 31")
	print("3.2 Метод Зейделя ---- 32")
	print("4 Метод вращений ---- 4")
	print("5 QR - алгоритм ---- 5")
	answer = input()
	if int(answer) == 1:
		print("LU-разложение")
		LU()
		
	if int(answer) == 2:
		print("Метод прогонки")
		methodOfRun()
	
	if int(answer) == 31:
		print("Метод простых итераций")
		methodOfSimpleIterations()

	if int(answer) == 32:
		print("Метод Зейделя")
		methodOfZeidel()

	if int(answer) == 4:
		print("Метод вращений")
		methodOfRotation()

	if int(answer) == 5:
		print("QR - алгоритм")
		methodQR()
		
	menu()

#solutionWithLU()
menu()