# SVM implements

import numpy as np
from sympy import * 

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
		# for i in range(self.n):
		# 	difl = [difl[j].subs(self.an[i], self.an[i]**2) for j in range(len(difl))]
		asol = solve(difl, self.an + [beta])

		for j, ai in enumerate(self.an):
			if asol[ai] < 0: 
				print(ai, asol[ai])
				subLt = Lt.subs(ai, 0.0)
				subdifl = []
				for i in range(self.n): 
					if i != j: subdifl.append(diff(subLt, self.an[i]))
				subdifl.append(diff(subLt, beta))
				nonnegaAsol = solve(subdifl, self.an[0:j] + self.an[j+1:len(asol)] + [beta])
				print(subdifl)
				print(nonnegaAsol)
				print('ajs:', self.an[0:j] + self.an[j+1: len(asol)])
		return asol

	def solve(self):
		asol = self.dualsolve()
		for ai in asol:
			print(asol[ai])

if __name__ == "__main__":
	
	xt = np.array([[3,3],[4,3],[1,1]])
	yt = np.array([1, 1, -1])

	svmsel = SVM(xt, yt)
	print(svmsel.solve())




