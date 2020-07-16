# SVM implements using SMO algorithm

import numpy as np

class SVM_SMO():

	def __init__(self, xs: np.ndarray, ys: np.ndarray):
		self.xs = xs # input data
		self.ys = ys # labels
		self.n = len(xs) # input data size
		self.k = 0 # Iteration count
		self.an = np.zeros(self.n) # Lagrangian parameters
		self.En = np.zeros(self.n) # difference
		self.yigxi = np.zeros(self.n)
		self.b = 0 # threshold
		self.epsi = 0.001 # Precision
		self.C = 0.7 # Punishment parameter
		self.p = 2 # Polynomial power

		# initialize En and yigxi
		for i in range(self.n): 
			self.En[i] = self.g(self.xs[i]) - self.ys[i]
			self.yigxi[i] = self.ys[i]*(self.En[i] + self.ys[i])

	def polykernel(self, xi, xj):
		return (sum(xi*xj) + 1)**self.p

	def Gausskernel(self, xi, xj, sigma):
		return np.exp(-sum((xi-xj)**2)/2/sigma**2)

	def g(self, xi):
		K = np.array([self.polykernel(xj, xi) for xj in self.xs])
		return np.sum(self.an*self.ys*K)+self.b

	def constrain_a(self, a, L, H):
		mina = max(L, a)
		maxa = min(mina, H)
		return maxa

	def KKT_satisfy(self, i):
		yigxi = self.yigxi[i]
		if self.an[i] == 0: return yigxi >= 1 - self.epsi
		elif (self.an[i] > 0 and self.an[i] < self.C): return (yigxi <= 1 + self.epsi and yigxi >= 1 - self.epsi)
		elif self.an[i] == self.C: return yigxi <= 1 + self.epsi

	def KKT_bias(self, i):
		return abs(self.yigxi[i] - 1)

	def outer_loop(self):
		asel = []
		aasel = []
		akttbias = []
		aakttbias = []
		for i in range(self.n):
			if (self.an[i] - self.C) > self.epsi or self.an[i] < -self.epsi:
				if not self.KKT_satisfy(i):
				    asel.append(i)
				    akttbias.append(self.KKT_bias(i))
			elif not self.KKT_satisfy(i): 
				aasel.append(i)
				aakttbias.append(self.KKT_bias(i))
		return asel, aasel

	def inner_loop(self, i):
		E1 = self.En[i]
		maxdE = 0
		a2sel = []
		# for j in range(self.n):
		# 	if j != i and self.En[j] != 0:
		# 		E2 = self.En[j]
		# 		if abs(E1-E2) > maxdE: 
		# 			maxdE = abs(E1-E2)
		# 			a2sel = j
		# return a2sel
		j = i
		while j == i: j=np.random.randint(0,self.n)
		return j

	def Stop_cond(self):
		if abs(np.sum(self.an*self.ys) - 0) <= self.epsi: return False
		for i in range(self.n):
			if (self.an[i] - self.C) > self.epsi or self.an[i] < -self.epsi: return False
			if not self.KKT_satisfy(i): return False
		return True

	def update_aiaj(self, i, j):
		E1 = self.En[i]
		E2 = self.En[j]
		if self.ys[i] == self.ys[j]:
			L = max(0, self.an[j] + self.an[i] - self.C)
			H = min(self.C, self.an[j] + self.an[i])
		else:
			L = max(0, self.an[j] - self.an[i])
			H = min(self.C, self.C + self.an[j] - self.an[i])
		if L == H: 
			# print('L == H')
			return 0
		eta = self.polykernel(self.xs[i],self.xs[i]) + self.polykernel(self.xs[i],self.xs[i]) - 2*self.polykernel(self.xs[i],self.xs[j]) 
		if eta <= 0: 
			# print('eta <= 0')
			return 0
		olda1, olda2 = self.an[i], self.an[j]
		self.an[j] += (E1 - E2) * self.ys[j] / eta
		self.an[j] = self.constrain_a(self.an[j], L, H)
		if abs(self.an[j]-olda2) < self.epsi:
			# print('a[j] does not have enough move')
			return 0

		self.an[i] += self.ys[i]*self.ys[j]*(olda2 - self.an[j])

		b1new = (-self.En[i] - self.ys[i] * self.polykernel(self.xs[i], self.xs[i]) * (self.an[i] - olda1) 
				- self.ys[j] * self.polykernel(self.xs[j], self.xs[i]) * (self.an[j] - olda2) + self.b)
		b2new = (-self.En[j] - self.ys[i] * self.polykernel(self.xs[i], self.xs[j]) * (self.an[i] - olda1) 
				- self.ys[j] * self.polykernel(self.xs[j], self.xs[j]) * (self.an[j] - olda2) + self.b)
		self.b = (b1new + b2new)*0.5
		# print('a[j] does have enough move')
		return 1

	def update_E(self):
		for i in range(self.n): 
			self.En[i] = self.g(self.xs[i]) - self.ys[i]
			self.yigxi[i] = self.ys[i]*(self.En[i] + self.ys[i])

	def solve(self):
		aupdate = 0
		while self.k < 200:
			aunbon, afull = self.outer_loop()
			if len(aunbon) > 0 and aupdate > 0: 
				aupdate = 0
				acheck = aunbon
			else: 
				acheck = afull
				aupdate = 0

			for i in acheck:
				j = self.inner_loop(i)
				aupdate += self.update_aiaj(i,j)
				if self.Stop_cond(): return self.an, self.b
				self.update_E()
			self.k += 1
			print("Iteration:", self.k, "number of a updated:", aupdate)

		return self.an, self.b


if __name__ == "__main__":
	
	data = np.loadtxt('SVM_training_data.txt')
	data = np.transpose(data)
	xs = np.transpose(data[0:2])
	ys = data[2]
	svm1 = SVM_SMO(xs, ys)
	an, b = svm1.solve()

# 画出分类结果
# 加入测试方法
