#-*-coding:utf-8-*-
import numpy as np

class perceptron:

	def __init__(self, w: np.ndarray, b: float, x: np.ndarray, y: np.ndarray, eta: float, itera: int):
		self.w = w
		self.b = b
		self.x = x
		self.y = y
		self.eta = eta
		self.itera = itera

	def distosurface(self,w,b,x,y):
		'''Not used in real calculation, just to show how to calculate the distance to a super surface'''
		disw = 0
		for i in w:
			disw += i**2
		disw = disw**0.5
		return -y*(w*x+b)/disw

	def Lossfunc(self,x,y,w,b):
		totloss = -y*(w*x+b)
		return sum(totloss)

	def diffLoss(self,x,y):
		dw = -x*y
		db = -y
		return dw, db

	def solve(self):
		nsamp = len(self.x)
		for i in range(self.itera):
			k = np.random.randint(nsamp)
			if self.Lossfunc(self.x[k], self.y[k], self.w, self.b) > 0:
				dw, db = self.diffLoss(self.x[k], self.y[k])
				self.w -= self.eta*dw
				self.b -= self.eta*db
				print('i=', i, 'Loss Function update:', self.Lossfunc(self.x[k], self.y[k], self.w, self.b))
		return self.w, self.b

if __name__ == "__main__":
	# Training data set
	x = np.array([[1,4,6],[6,2,7],[-1,-5,8],[6,9,0],[-4,-7,8],[10,-2,3],[5,0,0],[7,7,9],[1,5,2],[-6,7,0]])
	y = np.array([1,1,-1,1,-1,-1,1,1,1,-1])
	# Initial guess of w and b:
	w = np.array([1.1,2.0,3.0])
	b = 1.0
	eta = 0.01
	itera = 1000

	bisel = perceptron(w, b, x, y, eta, itera)
	print(bisel.solve())
