import numpy as np 
import matplotlib.pyplot as plt

def random_normalized(d1, d2):
    x = np.random.random((d1, d2))
    return x / x.sum(axis=1, keepdims=True)

class HMM:
    def __init__(self, M):
        self.M = M

    def to_onehot(self, x):
        K, T = max(x)+1, len(x)
        x_onehot = np.zeros((T, K))
        for i in range(T):
            x_onehot[i][x[i]] = 1.0
        return x_onehot



    def fit(self, X, n_iter = 50):
        # random initialization
        np.random.seed(123)
        N = len(X)
        Tmax = max(max(x) for x in X) + 1
        self.Pi = np.ones(self.M) / self.M
        self.A = random_normalized(self.M, self.M)
        self.B = random_normalized(self.M, Tmax)

        costs = []
        for it in range(n_iter):
            if it%5 == 0: print('Iteration: ', it)
            alphas = []
            betas = []
            P = np.zeros(N)
            for n in range(N):
                x = X[n]
                T = len(x)
                alpha = np.zeros((T, self.M))
                beta = np.zeros((T, self.M))
                alpha[0,:] = self.Pi*self.B[:,x[0]]
                beta[-1,:] = 1.0
                for i in range(1,T):
                    alpha[i] = self.B[:,x[i]]*((alpha[i-1].dot(self.A)))
                    beta[T-i-1] = (self.A.dot(self.B[:,x[T-i]]*beta[T-i]))
                P[n] = np.sum(alpha[-1,:])
                alphas.append(alpha)
                betas.append(beta)

            assert(np.all(P>0))
            cost = np.sum(np.log(P))
            costs.append(cost)

            self.Pi = np.sum((alphas[n][0] * betas[n][0])/P[n] for n in range(N)) / N
            numeA, numeB, denoA, denoB = 0, 0, 0, 0
            for n in range(N):
                x = X[n]
                x_onehot = self.to_onehot(x)
                numeA += ((alphas[n][:-1]).T.dot((self.B[:,x[1:]]).T*betas[n][1:]) * self.A)/P[n]
                denoA += (np.sum(alphas[n][:-1]*betas[n][:-1], axis = 0))/P[n]
                numeB += ((alphas[n]*betas[n]).T.dot(x_onehot))/P[n]
                denoB += (np.sum(alphas[n]*betas[n], axis = 0))/P[n]
            #print(numeA.shape, denoA.shape, numeB.shape, denoB.shape)
            self.A = numeA/(denoA.reshape(-1,1))
            self.B = numeB/(denoB.reshape(-1,1))
            # print(it)
            # print(numeA)
            # print(numeB)
            # print(denoA.reshape(-1,1))
            # print(denoB.reshape(-1,1))
            # print("A:", self.A)
            # print("B:", self.B)
            # print("Pi:", self.Pi)

        print("A:", self.A)
        print("B:", self.B)
        print("Pi:", self.Pi)

        plt.plot(costs)
        plt.show()


    def likelihood(self, x):
        # returns log P(x | model)
        # using the forward part of the forward-backward algorithm
        T = len(x)
        alpha = np.zeros((T, self.M))
        alpha[0] = self.Pi*self.B[:,x[0]]
        for i in range(1, T):
            alpha[i] = self.B[:,x[i]]*((alpha[i-1].dot(self.A)))
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
        delta[0] = self.Pi*self.B[:,x[0]]
        for t in range(1, T):
            for j in range(self.M):
                delta[t,j] = np.max(delta[t-1]*self.A[:,j]) * self.B[j, x[t]]
                psi[t,j] = np.argmax(delta[t-1]*self.A[:,j])

        # backtrack
        states = np.zeros(T, dtype=np.int32)
        states[T-1] = np.argmax(delta[T-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        return states



if __name__ == '__main__':

    X = []
    for line in open('coin_data.txt'):
        # 1 for H, 0 for T
        x = [1 if e == 'H' else 0 for e in line.rstrip()]
        X.append(x)

    hmmmodel = HMM(2)
    hmmmodel.fit(X)

    L = hmmmodel.log_likelihood_multi(X).sum()
    print("LL with fitted params:", L)

    # try true values
    hmmmodel.Pi = np.array([0.5, 0.5])
    hmmmodel.A = np.array([[0.1, 0.9], [0.8, 0.2]])
    hmmmodel.B = np.array([[0.6, 0.4], [0.3, 0.7]])
    L = hmmmodel.log_likelihood_multi(X).sum()
    print("LL with true params:", L)

    # try viterbi
    print("Best state sequence for:", X[0])
    print(hmmmodel.predict_sequence(X[0]))
    # X_sin = np.sin(np.linspace(1,20,10)) + 0.05*np.random.randn(10)
    # X_train = [X_sin]

    # hmmmodel = HMM(3)

    # hmmmodel.fit(X_train, 5)



## Broadcast check
    # t1 = np.random.randn(2).reshape(2,1)
    # t2 = np.random.randn(6).reshape(2, 3)
    # print(t1.T)
    # print(t2)
    # print(t1.T.dot(t2))
