#-*-coding:utf-8-*-
import numpy as np

class dual_perceptron:

	def __init__(self, a: np.ndarray, b: float, x: np.ndarray,
	 y: np.ndarray, eta: float, itera: int):
	    self.a = a
	    self.b = b
	    self.x = x
	    self.y = y
	    self.eta = eta
	    self.itera = itera

	def Lossfunc(self, x, y, a, b):
		totloss = sum(y*(a*x)+b*y)
		return totloss

	def grammat(self, x1s, x2s):
		x1s = np.asmatrix(x1s)
		x2s = np.asmatrix(x2s)
		return np.asarray(np.dot(x1s, x2s.T))

	def solve(self):
		nsamp = len(self.x)
		gramx = self.grammat(self.x, self.x)
		gramy = self.grammat(self.y, self.y)
		for i in range(self.itera):
			k = np.random.randint(nsamp)
			if self.Lossfunc(gramx[k], gramy[k], self.a, self.b) <= 0:
				self.a[k] += self.eta
				self.b += self.eta*self.y[k][0]
				print('i=', i, 'Loss Function update:', self.Lossfunc(gramx[k], gramy[k], self.a, self.b))
		return a, b

if __name__ == "__main__":
	## Training data set
	# x = np.array([[1,4,6],[6,2,7],[-1,-5,8],[6,9,0],[-4,-7,8],[10,-2,3],[5,0,0],[7,7,9],[1,5,2],[-6,7,0]])
	# y = np.array([[1],[1],[-1],[1],[-1],[-1],[1],[1],[1],[-1]])

	x = [[3,3],[4,3],[1,1]]
	y = [[1],[1],[-1]]

	# Initial guess of a and b:
	a = np.array([0.0]*len(x))
	b = 0.0
	eta = 1.0
	itera = 100

	#Training
	bisel = dual_perceptron(a, b, x, y, eta, itera)
	af, bf = bisel.solve()
	print(af, bf)
