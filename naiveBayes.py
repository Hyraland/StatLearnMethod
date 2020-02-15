import numpy as np

class selector:

	def naiveBayes(self, c: np.ndarray, xt: np.ndarray,
	 yt: np.ndarray, x: np.ndarray, condlambda: float):

		self.n = len(yt)

		def Pprior(y, ck):
			y_ck = 0
			for i in y:
				if i == ck: y_ck += 1
			return y_ck/self.n

		def condProp(ajl, x_j, y, ck, lamb):
			condp = 0
			y_p = 0
			sj = len(set(x_j))
			for i, j in zip(x_j, y):
				if j == ck:
					y_p += 1
					if i == ajl:
						condp += 1
			return (condp+lamb)/(y_p+sj*lamb)

		def postProp(xt, yt, x, ck, lamb):
			n = len(xt)
			nl = len(xt[0])
			condp = 1
			pprior = Pprior(yt, ck)
			for i in range(nl):
			    condp *= condProp(x[i], xt[:, i], yt, ck, lamb)
			return pprior*condp

		def classifyx(xt, yt, x, c, lamb):
			maxpost = 0
			maxpc = 0
			for ck in c:
				postp = postProp(xt, yt, x, ck, lamb)
				if postp > maxpost: maxpost, maxpc = postp, ck
			return maxpc

		return classifyx(xt, yt, x, c, condlambda)

if __name__ == "__main__":
	
	c = np.array([1,-1])
	xt = np.array([[1,1], [1,2], [1,2], [1,1], [1,1], [2,1], [2,2], [2,2], [2,3], [2,3], [3,3], [3,2], [3,2], [3,3], [3,3]])
	yt = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
	lamb = 1.0

	x = np.array([2,1])

	bisel = selector()
	print(bisel.naiveBayes(c, xt, yt, x, lamb))



