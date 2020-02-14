#-*-coding:utf-8-*-
import numpy as np

class biselector:

	def dual_perceptron(self, a: np.ndarray, b: float, x: np.ndarray,
	 y: np.ndarray, eta: float, itera: int):

		def Lossfunc(x,y,a,b):
			totloss = sum(y*(a*x)+b*y)
			return totloss

		def grammat(x1s, x2s):
			x1s = np.asmatrix(x1s)
			x2s = np.asmatrix(x2s)
			return np.asarray(np.dot(x1s, x2s.T))

		nsamp = len(x)
		gramx = grammat(x, x)
		gramy = grammat(y, y)
		for i in range(itera):
			k = np.random.randint(nsamp)
			if Lossfunc(gramx[k], gramy[k], a, b) <= 0:
				a[k] += eta
				b += eta*y[k][0]
				print('i=', i, 'Loss Function update:', Lossfunc(gramx[k],gramy[k],a,b))
		return a, b


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
bisel = biselector()
af, bf = bisel.dual_perceptron(a, b, x, y, eta, itera)
print(af, bf)
