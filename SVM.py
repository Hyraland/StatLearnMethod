# SVM implements

import numpy as np
from sympy import * 

# #设置变量
# x1 = symbols("x1")
# x2 = symbols("x2")
# alpha = symbols("alpha")
# beta = symbols("beta")
 
# #构造拉格朗日等式
# L = 10 - x1*x1 - x2*x2 + alpha * (x1*x1 - x2) + beta * (x1 + x2)
# print(type(L), type(x1)) #<class 'sympy.core.add.Add'> <class 'sympy.core.symbol.Symbol'>
# #求导，构造KKT条件
# difyL_x1 = diff(L, x1)  #对变量x1求导
# difyL_x2 = diff(L, x2)  #对变量x2求导
# difyL_beta = diff(L, beta)  #对乘子beta求导
# dualCpt = alpha * (x1 * x1 - x2)  #对偶互补条件
 
# #求解KKT等式
# aa = solve([difyL_x1, difyL_x2, difyL_beta, dualCpt], [x1, x2, alpha, beta])
 
# #打印结果，还需验证alpha>=0和不等式约束<=0
# for i in aa:
#     if i[2] >= 0:
#         if (i[0]**2 - i[1]) <= 0:
#             print(i)

class SVM():

	def __init__(self, xs: np.ndarray, ys: np.ndarray):
		self.xs = xs
		self.ys = ys
		self.n = len(xs)
		self.an = [symbols('a'+str(i)) for i in range(self.n)]
		self.cn = [symbols('c'+str(i)) for i in range(self.n)] # cn为约束条件ai>=0 for i in (1,n)的约束条件的乘子

	def dualsolve(self):
		beta = symbols("beta")
		L, Lb, Lc, Lac = 0, 0, 0, 0 # L,与Lb为SVM对偶约束函数中的组成部分， Lc为约束条件，Lac为另一约束条件ai>=0 for i in (1,n)的约束条件
		for i in range(self.n):
			Lc += self.an[i]*self.ys[i]
			Lb += self.an[i]
			Lac += self.cn[i] * self.an[i]
			for j in range(self.n):
				L += self.an[i]*self.an[j]*self.ys[i]*self.ys[j]*(np.dot(self.xs[i],self.xs[j]))

		Lt = 0.5*L - Lb + Lc * beta #+ Lac
		difl = []
		dualkkt = []
		for i in range(self.n): difl.append(diff(Lt, self.an[i]))
		difl.append(diff(Lt, beta))
		for i in range(self.n): difl.append(diff(Lt, self.cn[i]))
		for i in range(self.n): dualkkt.append(self.cn[i] * self.an[i])
		# asol = solve(difl + dualkkt, self.an + [beta] + self.cn)
		asol = solve(difl, self.an + [beta])
		return asol

if __name__ == "__main__":
	
	xt = np.array([[3,3],[4,3],[1,1]])
	yt = np.array([1, 1, -1])

	svmsel = SVM(xt, yt)
	print(svmsel.dualsolve())




