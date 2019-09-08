import math
import copy

#2_!
def f(x):
	return math.log(x+1) - 2*x + 0.5

def derivativeF(x):
	return 1/(x+1) - 2
 
def methodOfNewton():
	print("введите точность вычислений")
	accuracy = float(input())
	x = 0.3
	xPrev = None
	while xPrev == None or abs(x - xPrev) > accuracy:
		xPrev = x
		x = x - f(x) / derivativeF(x)
	print(x)

def fi(x):
	return (math.log(x+1) + 0.5) / 2

def methodOfSimpleIterations():
	print("введите точность вычислений")
	accuracy = float(input())
	x = 0.8
	xPrev = None
	while xPrev == None or abs(x - xPrev) > accuracy:
		xPrev = x
		x = fi(x)
	print(x)

#2_2
def f1(x1,x2):
	return 0.25*x1*x1 + x2*x2 - 1

def f2(x1,x2):
	return 2*x2 - math.exp(x1) - x1

def f1dx1(x1,x2):
	return 0.5*x1

def f1dx2(x1,x2):
	return 2*x2

def f2dx1(x1,x2):
	return -math.exp(x1) - 1

def f2dx2(x1,x2):
	return 2

def det(a):
	return a[0][0] * a[1][1] - a[0][1] * a[1][0]

def getNorm(x1,x2):
	max = 0
	for i in range(len(x1)):
		if abs(x1[i] - x2[i]) > max:
			max = abs(x1[i] - x2[i])
	return max


def methodOfNewtonForSystem():
	print("введите точность вычислений")
	accuracy = float(input())
	x = [0.6, 1]
	xPrev = []
	while len(xPrev) == 0 or getNorm(x, xPrev) > accuracy:
		
		xPrev = copy.deepcopy(x)
		x[0] = x[0] - det([[f1(xPrev[0], xPrev[1]),f1dx2(xPrev[0], xPrev[1])],[f2(xPrev[0], xPrev[1]), f2dx2(xPrev[0], xPrev[1])]]) / det([[f1dx1(xPrev[0], xPrev[1]), f1dx2(xPrev[0], xPrev[1])],[f2dx1(xPrev[0], xPrev[1]), f2dx2(xPrev[0], xPrev[1])]])
		x[1] = x[1] - det([[f1dx1(xPrev[0], xPrev[1]),f1(xPrev[0], xPrev[1])],[f2dx1(xPrev[0], xPrev[1]), f2(xPrev[0], xPrev[1])]]) / det([[f1dx1(xPrev[0], xPrev[1]), f1dx2(xPrev[0], xPrev[1])],[f2dx1(xPrev[0], xPrev[1]), f2dx2(xPrev[0], xPrev[1])]])
	print(x)

def fi1(x1,x2):
	return 2*x2 - math.exp(x1)

def fi2(x1,x2):
	return math.sqrt(abs(1-0.25*x1*x1))

def methodOfSimpleIterationsForSystem():
	print("введите точность вычислений")
	accuracy = float(input())
	x = [0.6, 1]
	xPrev = []
	k = 0
	while len(xPrev) == 0 or getNorm(x, xPrev)  > accuracy:
		k += 1
		xPrev = copy.deepcopy(x)
		x[0] = fi1(xPrev[0],xPrev[1])
		x[1] = fi2(xPrev[0],xPrev[1])
		
	print(x)
	print(k)

def menu():
	print("1.1 Метод Ньютона ---- 1")
	print("1.2 Метод простых итераций ---- 12")
	print("2.1 Метод Ньютона для системы уравнений ---- 21")
	print("2.2 Метод простых итераций для системы уравнений ---- 22")
	a = input()
	if a == '1':
		print("Метод Ньютона")
		methodOfNewton()
		menu()

	if a == '12':
		print("Метод простых итераций")	
		methodOfSimpleIterations()
		menu()

	if a == '21':
		print("Метод Ньютона для системы уравнений")
		methodOfNewtonForSystem()
		menu()

	if a == '22':
		print("Метод простых итераций для системы уравнений")
		methodOfSimpleIterationsForSystem()
		menu()

menu()