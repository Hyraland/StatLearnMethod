import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from generate_c import get_signals, big_init, simple_init

import wave

def random_normalized(d1, d2):
    x = np.random.random((d1, d2))
    return x / x.sum(axis=1, keepdims=True)

class HMM:
    def __init__(self, M, K):
        self.M = M
        self.K = K

    def to_onehot(self, x):
        K, T = max(x)+1, len(x)
        x_onehot = np.zeros((T, K))
        for i in range(T):
            x_onehot[i][x[i]] = 1.0
        return x_onehot



    def fit(self, X, n_iter = 30):
        # random initialization
        np.random.seed(153)
        N = len(X)
        D = X[0].shape[1]
        Tmax = max(max(x) for x in X) + 1
        self.Pi = np.ones(self.M) / self.M
        self.A = random_normalized(self.M, self.M)
        self.R = np.ones((self.M, self.K))/self.K
        self.mu = np.zeros((self.M, self.K, D)) # M hidden states, K gaussians, D dim for each gaussian
        for i in range(self.M):
            for j in range(self.K):
                n = np.random.choice(N)
                t = np.random.choice(len(X[n]))
                self.mu[i][j] = X[n][t]

        self.sigma = np.zeros((self.M, self.K, D, D))
        for i in range(self.M):
            for j in range(self.K):
                self.sigma[i][j] = np.eye(D)

        costs = []
        for it in range(n_iter):
            if it%5 == 0: print('Iteration: ', it)
            alphas = []
            betas = []
            gammas = []
            Bs = []
            P = np.zeros(N)
            for n in range(N):
                x = X[n]
                T = len(x)

                B = np.zeros((self.M, T))
                components = np.zeros((self.M, self.K, T))
                for i in range(self.M):
                    for j in range(self.K):
                        components[i,j,:] = self.R[i,j] * mvn.pdf(x, self.mu[i][j], self.sigma[i][j])
                    B[i] = np.sum(components[i], axis = 0)
                Bs.append(B)

                alpha = np.zeros((T, self.M))
                beta = np.zeros((T, self.M))
                alpha[0,:] = self.Pi*B[:,0]
                beta[-1,:] = 1.0
                for i in range(1,T):
                    alpha[i] = B[:,i]*((alpha[i-1].dot(self.A)))
                    beta[T-i-1] = (self.A.dot(B[:,T-i]*beta[T-i]))
                P[n] = np.sum(alpha[-1,:])
                alphas.append(alpha)
                betas.append(beta)

                gamma = np.zeros((T, self.M, self.K))
                for t in range(T):
                    ab = np.sum(alpha[t,:]*beta[t,:])
                    for k in range(self.K):
                        gamma[t,:,k] = alpha[t,:]*beta[t,:]/(B[:,t]*ab) *components[:,k,t]
                gammas.append(gamma)

            cost = np.sum(np.log(P))
            costs.append(cost)

            self.Pi = np.sum((alphas[n][0] * betas[n][0])/P[n] for n in range(N)) / N
            numeA, denoA, numeR, denoR, numemu = 0, 0, 0, 0, 0
            numemu = np.zeros((self.M, self.K, D))
            numesig = np.zeros((self.M, self.K, D, D))
            for n in range(N):
                x = X[n]
                B = Bs[n]
                numeA += ((alphas[n][:-1]).T.dot((B[:,1:]).T*betas[n][1:]) * self.A)/P[n]
                denoA += (np.sum(alphas[n][:-1]*betas[n][:-1], axis = 0))/P[n]
                numeR += np.sum(gammas[n], axis = 0)/P[n]
                for i in range(self.M):
                    for j in range(self.K):
                        numemu[i,j] += x.T.dot(gammas[n][:,i,j])/P[n]
                        for t in range(T):
                            #print("before update", n,i,j,numesig[i,j].shape, np.outer(x[t] - self.mu[i,j], x[t] - self.mu[i,j]).shape)
                            numesig[i,j] += gamma[t,i,j] * np.outer(x[t] - self.mu[i,j], x[t] - self.mu[i,j])/P[n]
                # if n==0 and it == 0:
                #     print(numeA)
                #     print(numeR)
                #     print(denoA)
                #     print(numemu)
                #     print(numesig)
                #     print(gammas[n])
                #     print(np.sum(gammas[n], axis = 0))
               
            denoR = np.sum(numeR, axis = 1)
            self.A = numeA/(denoA.reshape(-1,1))
            for i in range(self.M):
                self.R[i] = numeR[i]/denoR[i]
                for j in range(self.K):            
                    self.mu[i,j] = numemu[i,j]/numeR[i,j]
                    self.sigma[i,j] = numesig[i,j]/numeR[i,j]

            print("A:", self.A)
            print("R:", self.R)
            print("mu:", self.mu)
            print("sigma:", self.sigma)
            print("Pi:", self.Pi)

        plt.plot(costs)
        plt.show()


    def likelihood(self, x):
        # returns log P(x | model)
        # using the forward part of the forward-backward algorithm
        T = len(x)
        alpha = np.zeros((T, self.M))
        B = np.zeros((self.M, T))
        components = np.zeros((self.M, self.K, T))
        for i in range(self.M):
            for j in range(self.K):
                components[i,j,:] = self.R[i][j] * mvn.pdf(x, self.mu[i][j], self.sigma[i][j])
                B[i] = np.sum(components[i], axis = 0)

        alpha[0] = self.Pi*B[:,0]
        for i in range(1, T):
            alpha[i] = B[:,i]*((alpha[i-1].dot(self.A)))
        return alpha[-1].sum()

    def likelihood_multi(self, X):
        return np.array([self.likelihood(x) for x in X])

    def log_likelihood_multi(self, X):
        return np.log(self.likelihood_multi(X))

    def predict_sequence(self, x):
        # returns the most likely state sequence given observed sequence x
        # using the Viterbi algorithm
        T = len(x)
        delta = np.zeros((T, self.M))
        psi = np.zeros((T, self.M))

        B = np.zeros((self.M, T))
        components = np.zeros((self.M, self.K, T))
        for i in range(self.M):
            for j in range(self.K):
                components[i,j,:] = self.R[i][j] * mvn.pdf(x, self.mu[i][j], self.sigma[i][j])
                B[i] = np.sum(components[i], axis = 0)

        delta[0] = self.Pi*B[:,0]
        for t in range(1, T):
            for j in range(self.M):
                delta[t,j] = np.max(delta[t-1]*self.A[:,j]) * B[j, t]
                psi[t,j] = np.argmax(delta[t-1]*self.A[:,j])

        # backtrack
        states = np.zeros(T, dtype=np.int32)
        states[T-1] = np.argmax(delta[T-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        return states

    def set(self, Pi, A, R, mu, sigma):
        self.Pi = Pi
        self.A = A
        self.R = R
        self.mu = mu
        self.sigma = sigma
        self.M, self.K = R.shape[0], R.shape[1]



def real_signal():
    spf = wave.open('helloworld.wav', 'r')

    #Extract Raw Audio from Wav File
    # If you right-click on the file and go to "Get Info", you can see:
    # sampling rate = 16000 Hz
    # bits per sample = 16
    # The first is quantization in time
    # The second is quantization in amplitude
    # We also do this for images!
    # 2^16 = 65536 is how many different sound levels we have
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, 'Int16')
    T = len(signal)

    hmm = HMM(10)
    hmm.fit(signal.reshape(1, T))


def fake_signal(init=simple_init):
    signals = get_signals(N=1, T=10, init=init)
    for signal in signals:
        for d in range(signal.shape[1]):
            plt.plot(signal[:,d])
    plt.show()

    hmm = HMM(2, 2)
    hmm.fit(signals)
    L = hmm.log_likelihood_multi(signals).sum()
    print("LL for fitted params:", L)
    plt.plot(hmm.predict_sequence(signals[0]))
    plt.show()
    # test in actual params
    _, _, _, pi, A, R, mu, sigma = init()
    hmm.set(pi, A, R, mu, sigma)
    L = hmm.log_likelihood_multi(signals).sum()
    print("LL for actual params:", L)

    # print most likely state sequence
    print("Most likely state sequence for initial observation:")
    print(hmm.predict_sequence(signals[0]))

if __name__ == '__main__':
    # real_signal() # will break
    fake_signal(init=simple_init)
    # fake_signal(init=big_init) # will break

