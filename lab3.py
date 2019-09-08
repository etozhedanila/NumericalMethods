import math
from matrix import * 
import copy

#3_1
def f(x):
	if type(x) == float or type(x) == int:
		return math.cos(x) + x
	if len(x) == 1:
		return math.cos(x[0]) + x[0]
	else: 
		return (f(x[:-1]) - f(x[1:])) / (x[0] - x[-1])



def w(tmpX, x):
	result = 1
	for i in range(len(x)):
		if tmpX != x[i]:
			result *= tmpX - x[i]

	return result

def polynomialOfLagrange():
	x1 = [0, math.pi / 6, 2*math.pi / 6, 3*math.pi / 6]
	x2 = [0, math.pi / 6, math.pi / 4, math.pi / 2]
	x = 1
	l1 = 0
	l2 = 0
	for i in range(len(x1)):
		l1 += f(x1[i]) * w(x,x1) / ((x - x1[i])*w(x1[i],x1) )
		l2 += f(x2[i]) * w(x,x2) / ((x - x2[i])*w(x2[i],x2) )
	print("Пункт а)")
	print("L(x) = ", l1)
	print("y(x) = ", f(x))
	print("Абсолютная погрешность: ", abs(f(x) - l1))

	print("Пункт б)")
	print("L(x) = ", l2)
	print("y(x) = ", f(x))
	print("Абсолютная погрешность: ", abs(f(x) - l2))





def polynomialOfNewton():
	x1 = [0, math.pi / 6, 2*math.pi / 6, 3*math.pi / 6]
	x2 = [0, math.pi / 6, math.pi / 4, math.pi / 2]
	x = 1
	p1 = f(x1[0])
	p2 = f(x2[0])
	tmp1 = 1 
	tmp2 = 1
	for i in range(1,len(x1)):
		
		tmp1 *= x - x1[i - 1]
		tmp2 *= x - x2[i - 1]
		p1 += tmp1 * f(x1[:i+1])
		p2 += tmp2 * f(x2[:i+1])

	print("Пункт а)")
	print("P(x) = ", p1)
	print("y(x) = ", f(x))
	print("Абсолютная погрешность: ", abs(f(x) - p1))

	print("Пункт б)")
	print("P(x) = ", p2)
	print("y(x) = ", f(x))
	print("Абсолютная погрешность: ", abs(f(x) - p2))

#3_2
def h(x2,x1):
	return x2-x1

def methodOfRun(d,m):
	p = []
	q = []
	
	for i in range(len(d)):
		row = m[i]
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

	x = [0]*len(d)
	x[-1] = q[-1]
	for i in range(len(d) - 2, -1, -1):
		x[i]=(p[i]*x[i+1]+q[i])
	
	return x

def buildSpline():
	x = [0, 1, 2, 3, 4]
	f = [1, 1.5403, 1.5839, 2.01, 3.3464]
	
	xPoint = 1.5
	
	d = []
	matrix = []
	for i in range(3):
		tmp = []
		d.append( 3*((f[i+2]-f[i+1]) / h(x[i+2],x[i+1]) - (f[i+1]-f[i]) / h(x[i+1],x[i])))
		if i == 0:
			tmp.append( 2 * ( h(x[1],x[0]) + h(x[2],x[1])) )
			tmp.append( h(x[2],x[1]) )
		if i == 1:
			tmp.append( h(x[2],x[1]) )
			tmp.append( 2 * (h(x[3],x[2]) + h(x[2],x[1])) )
			tmp.append( h(x[3],x[2]) )
		if i == 2:
			tmp.append( h(x[3],x[2]) )
			tmp.append( 2 * (h(x[3],x[2]) + h(x[4],x[3])) )
		matrix.append(tmp)

	c = methodOfRun(d,matrix)

	c.insert(0,0)
	a = []
	b = []
	d = []
	for i in range(1,len(x)):
		a.append(f[i-1])

		if i == len(x) - 1:

			d.append(-c[i-1] / (3*h(x[i],x[i-1])))
			b.append( (f[i] - f[i-1]) / h(x[i], x[i-1]) - 2 * c[i-1] * h(x[i],x[i-1]) / 3 )

		else:
			
			b.append( (f[i] - f[i-1]) / h(x[i], x[i-1]) - h(x[i],x[i-1]) * (c[i] + 2 * c[i-1]) / 3)
			d.append( (c[i] - c[i-1]) / (3 * h(x[i],x[i-1])) )
	
	print("a = ", a)
	print("b = ", b)
	print("c = ", c)
	print("d = ", d)

	result = 0
	for i in range(1, len(x)):
		if x[i-1] <= xPoint and xPoint <= x[i]:
			print(x[i-1],x[i])
			result = a[i-1] + b[i-1]*(xPoint - x[i-1]) + c[i-1]*(xPoint - x[i-1])*(xPoint - x[i-1]) + d[i-1] * (xPoint - x[i-1])*(xPoint - x[i-1])*(xPoint - x[i-1])
			

	print("f(x) = ", result)


#3_3
def LU(a):
	
	n = len(a) 
	
	u = [0] * n
	m = [[0] * n for i in range(n)]
	for k in range(n-1):	#k - номер шага
		for i in range(k+1, n):		#i - номер строки
			u[i] = a[i][k] / a[k][k]
			m[i][k] = u[i]
			for j in range(k, n):		#j - номер столбца
				a[i][j] = a[i][j] - u[i]*a[k][j]
	
	
	for i in range(n):
		m[i][i] = 1
		for j in range(i+1,n):
			m[i][j] = 0
		
	return m, a

def solutionWithLU(b, m):

	l, u = LU(copy.deepcopy(m))
	n = len(l)
	z = []
	
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

	return x

def leastSquareMethod():
	x = [-1, 0, 1, 2, 3, 4]
	y = [-0.4597, 1, 1.5403, 1.5839, 2.010, 3.3464]

	m = [[len(x), 0],[0, 0]]
	m3, m4 = 0, 0
	d = [0,0,0]
	for i in range(len(x)):
		m[0][1] += x[i]
		m[1][1] += x[i]*x[i]
		m3 += x[i]*x[i]*x[i]
		m4 += x[i]*x[i]*x[i]*x[i]

		d[0] += y[i]
		d[1] += y[i]*x[i]
		d[2] += y[i]*x[i]*x[i]
	m[1][0] = m[0][1]
	
	a = solutionWithLU(d[0:2],m)
	print("Приближающий многочлен первой степени F1(x) = ", a[0], " + ", a[1], "x")
	sum = 0
	for i in range(len(x)):
		sum += (a[0] + a[1] * x[i] - y[i]) * (a[0] + a[1] * x[i] - y[i])
	print("Сумма квадратов ошибок = ", sum)

	m[0].append(m[1][1])
	m[1].append(m3)
	m.append([m[1][1], m3, m4])
	
	a = solutionWithLU(d,m)
	print("Приближающий многочлен второй степени F2(x) = ", a[0], " + ", a[1], "x + ", a[2], "x^2")
	sum = 0
	for i in range(len(x)):
		sum += (a[0] + a[1] * x[i] + a[2]*x[i]*x[i] - y[i]) * (a[0] + a[1] * x[i] + a[2]*x[i]*x[i] - y[i])
	print("Сумма квадратов ошибок = ", sum)

#3_4
def calculateDerivative():
	x = [0.2, 0.5, 0.8, 1.1, 1.4]
	y = [12.906, 5.5273, 3.8777, 3.2692, 3.0319]
	point = 0.8
	firstDerivative = 0
	secondDerivative = 0
	for i in range(len(x) - 2):
		if x[i] <= point and point <= x[i+1]:
			print("Выбран отрезок  [", x[i], x[i+1], "]")
			firstDerivative = (y[i+1] - y[i]) / (x[i+1] - x[i]) + ( (y[i+2] - y[i+1]) / (x[i+2] - x[i+1]) - (y[i+1] - y[i]) / (x[i+1] - x[i]) ) / (x[i+2] - x[i]) * (2 * point - x[i] - x[i + 1])
			secondDerivative = 2 * ( (y[i+2] - y[i+1]) / (x[i+2] - x[i+1]) - (y[i+1] - y[i]) / (x[i+1] - x[i]) ) / (x[i+2] - x[i])

			break

	print("Первая производная в точке X* = ", firstDerivative)
	print("Вторая производная в точке X* = ", secondDerivative)

#3_5
def y(x):
	return x*x / (x*x*x - 27)

def calculateIntegralWithStep(h):
	print("Шаг = ",h)
	
	x0, xK = -2, 2
	rectangleResult = 0
	trapezeResult = y(x0) / 2
	simpsonResult = 0
	x = x0
	k = 1
	while x < xK:
		rectangleResult += y((x + x + h) / 2)
		trapezeResult += y(x)
		simpsonResult += y(x) * k
		if k == 4:
			k = 2
		else:
			k = 4
		x += h

	rectangleResult *= h
	print("Метод прямоугольника F = ", rectangleResult)

	trapezeResult += y(xK) / 2 - y(x0)
	trapezeResult *= h
	print("Метод трапеции F = ", trapezeResult)

	simpsonResult += y(xK)
	simpsonResult *= h / 3
	print("Метод Симпсона F = ", simpsonResult)

	return rectangleResult, trapezeResult, simpsonResult

def calculateIntegral():
	h1, h2 = 1, 0.5

	f1h1, f2h1, f3h1 = calculateIntegralWithStep(h1)
	f1h2, f2h2, f3h2 = calculateIntegralWithStep(h2)
	k = h2 / h1

	rrrResult1 = f1h1 + (f1h1 - f1h2) / (math.pow(k, 2) - 1)
	print("Метод Рунге-Ромберга-Ричардсона для прямоугольников", rrrResult1)

	rrrResult2 = f2h1 + (f2h1 - f2h2) / (math.pow(k, 2) - 1)
	print("Метод Рунге-Ромберга-Ричардсона для трапеций", rrrResult2)

	rrrResult3 = f3h1 + (f3h1 - f3h2) / (math.pow(k, 2) - 1)
	print("Метод Рунге-Ромберга-Ричардсона для Симпсона", rrrResult3)

	





def menu():
	print("1.1 Построить интерполяционный многочлен Лагранжа ---- 1")
	print("1.2 Построить интерполяционный многочлен Ньютона ---- 12")
	print("2 Построить кубический сплайн ---- 2")
	print("3 Метод наименьших квадратов ---- 3")
	print("4 Вычислить первую и вторую производную в точке ---- 4")
	print("5 Вычислить определенный интеграл ---- 5")
	
	a = input()
	if a == '1':
		print("Построить интерполяционный многочлен Лагранжа")
		polynomialOfLagrange()
		menu()

	if a == '12':
		print("Построить интерполяционный многочлен Ньютона")	
		polynomialOfNewton()
		menu()

	if a == '2':
		print("Построить кубический сплайн ")
		buildSpline()
		menu()

	if a == '3':
		print("Метод наименьших квадратов ")
		leastSquareMethod()
		menu()

	if a == '4':
		print("Вычислить первую и вторую производную в точке")
		calculateDerivative()
		menu()

	if a == '5':
		print("Вычислить определенный интеграл")
		calculateIntegral()
		menu()

menu()

