import numpy as np
import matplotlib.pyplot as plt

class soft_kmeans(object):
    def fit(self, x_train, K, beta, epoch):
        N, D = len(x_train), len(x_train[0])
        Y = np.zeros(N)
        c = np.random.choice(N,K)
        # initiate y
        y = x_train[c]
        rk = np.ones(N)
        tracky = np.ones((epoch,K))
        x_train = x_train.reshape(N,1,D)
        y = y.reshape(1,K,D)

        for e in range(epoch):
            rc = np.exp(-beta*self.dist(x_train, y))
                #rc0 = np.exp(-beta*self.dist(x_train, y))
                #print(x_train.shape, x_train[0].shape, y.shape)
                #print(x_train.reshape(1,N,D).shape, y.reshape(1,K,D).shape)

            Y = np.argmin(rc,axis=1)
            rk = np.min(rc,axis=1)/np.sum(rc)

            for j in range(K):
                # in case there are 0-sized categories
                if (len(rk[Y==j])>0):
                   y[0][j] = np.sum(rk[Y==j].reshape((len(rk[Y==j]), 1))*x_train.reshape(N,D)[Y==j],axis=0)/np.sum(rk[Y==j])

            tracky[e] = np.sum(y,axis=2)
            if e%20 == 0: print("Epoch:{:d}, newcenter at D1: {:f}".format(e,tracky[e][0]))

        return Y, tracky;

    def dist(self, x,y):
        return np.sqrt(np.sum((y-x)**2,axis=2))



def main():

    # Generate 2D data
    N = 400
    D = 2
    X1 = np.random.randn(N,D)
    X2 = np.random.randn(N,D)
    X3 = np.random.randn(N,D)
    sigma1, sigma2,sigma3, mean1x, mean1y, mean2x, mean2y, mean3x, mean3y = 1.5,2.0,1.5,3.0,8.0,7.5,2.5,0.0,0.0
    X1 = X1*sigma1+np.stack((np.ones(N)*mean1x, np.ones(N)*mean1y), axis=-1)
    X2 = X2*sigma2+np.stack((np.ones(N)*mean2x, np.ones(N)*mean2y), axis=-1)
    X3 = X3*sigma2+np.stack((np.ones(N)*mean3x, np.ones(N)*mean3y), axis=-1)
    X = np.concatenate((X1,X2,X3),axis=0)
    # plot the distribution of data
    plt.figure()
    plt.plot(X[:,0],X[:,1],'.')
    plt.show()
    # Seperate Train data and test data, not necessary in K-means
    x_train = X
    # x_test = X[:,500:]

    # fit
    K = 3
    beta = 0.1
    epoch = 200
    softkmeans = soft_kmeans()
    y_train, ytrack = softkmeans.fit(x_train, K, beta, epoch)

    # plot the prediction
    plt.figure()
    plt.scatter(X[:,0],X[:,1],c = y_train)
    plt.show()


if __name__=='__main__':
    main()

