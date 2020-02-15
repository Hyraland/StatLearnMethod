#-*-coding:utf-8-*-
import numpy as np

class biselector:

	def perceptron(self, w: np.ndarray, b: float, x: np.ndarray, y: np.ndarray, eta: float, itera: int):

		def distosurface(w,b,x,y):
			'''Not used in real calculation, just to show how to calculate the distance to a super surface'''
			disw = 0
			for i in w:
				disw += i**2
			disw = disw**0.5
			return -y*(w*x+b)/disw

		def Lossfunc(x,y,w,b):
			totloss = -y*(w*x+b)
			return sum(totloss)

		def diffLoss(x,y):
			dw = -x*y
			db = -y
			return dw, db

		nsamp = len(x)
		for i in range(itera):
			k = np.random.randint(nsamp)
			if Lossfunc(x[k], y[k], w, b) > 0:
				dw, db = diffLoss(x[k],y[k])
				w -= eta*dw
				b -= eta*db
				print('i=', i, 'Loss Function update:', Lossfunc(x[k],y[k],w,b))
		return w, b

if __name__ == "__main__":
	# Training data set
	x = np.array([[1,4,6],[6,2,7],[-1,-5,8],[6,9,0],[-4,-7,8],[10,-2,3],[5,0,0],[7,7,9],[1,5,2],[-6,7,0]])
	y = np.array([1,1,-1,1,-1,-1,1,1,1,-1])
	# Initial guess of w and b:
	w = np.array([1.1,2.0,3.0])
	b = 1.0
	eta = 0.01
	itera = 1000

	bisel = biselector()
	print(bisel.perceptron(w, b, x, y, eta, itera))
