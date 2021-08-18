import numpy as np
from sklearn.gaussian_process import kernels
from sklearn.gaussian_process import GaussianProcessRegressor

# X = [0.1, 0.3, 0.2, ]
# y = 
class ContinuumArmedBandit:
    def __init__(self, X, y, convergence_rate=1.0):
        self.X = X
        self.y = y
        self.gpr = GPR(self.X, self.y, convergence_rate=convergence_rate)
        self.N = self.X.shape[0]

        self.alpha = self.calc_alpha(self.gpr.K, self.gpr.noise_var, self.y)
        self.gamma = self.calc_gamma(self.gpr.K, self.gpr.noise_var, self.X)

    def select_action(self):
        x = self.get_x_best(self.X)
        return x
    
    def update(self, X, y)
        self.X = self.X.append(X)
        self.y = self.y.append(y)
        self.gpr = GPR(self.X, self.y, convergence_rate=convergence_rate)
        self.N = self.X.shape[0]

        self.alpha = self.calc_alpha(self.gpr.K, self.gpr.noise_var, self.y)
        self.gamma = self.calc_gamma(self.gpr.K, self.gpr.noise_var, self.X)

    def get_x_best(self, X):
        merit_best = 0
        x_best = None
        for j in range(self.N):
            x = X[j]
            temp_x = x
            for i in range(10):
                s = self.get_s(temp_x, X)
                temp_x = temp_x + x
            x_merit = self.get_merit(x, X)
            if x_merit > merit_best:
                x_best = x
                merit_best = x_merit
        return x_best

    def get_s(self, x, X):
        der_mean = self.get_derivative_mean(x, X)
        der_std = self.get_derivative_std(x, X)
        s = der_mean + der_std
        return s
    
    def get_derivative_std(self, x, X):
        der_std = 0
        W = self.gpr.W
        k_prime = self.gpr.k
        std = self.get_std(x, X)
        for i in range(self.N):
            for j in range(self.N):
                der_std += (2.0 / std) * self.gamma[i,j] * W.dot((X[i] + X[j]) / 2.0 - x) * k_prime(x, (X[i] + X[j]) / 2.0) 
        return der_std


    def get_derivative_mean(self, x, X):
        der_mean = 0
        W = self.gpr.W
        k = self.gpr.k
        for i in range(self.N):
            der_mean += W.dot(X[i] - x) * k(x, X[i]) * self.alpha[i]
        return der_mean


    def get_merit(self, x, X):
        mean = self.get_mean(x, X)
        var = self.get_std(x, X)
        merit = mean + var
        return merit
    
    def predict(self, x_arr):
        val_pred = []
        val_var = []
        for x in x_arr:
            mean = self.get_mean(x, self.X)
            var = self.get_std(x, self.X)
            val_pred.append(mean)
            val_var.append(var)
        return val_pred, val_var

    def get_mean(self, x, X):
        mean = 0
        k = self.gpr.k
        for j in range(self.N):
            mean += k(x,X[j]) * self.alpha[j]
        return mean

    def get_std(self, x, X):
        k = self.gpr.k
        k_prime = self.gpr.k_prime
        kxx = k(x,x)
        a = 0
        for i in range(self.N):
            for j in range(self.N):
                a += k_prime(x, 0.5 * (X[i] + X[j])) * self.gamma[i,j]
        var = np.sqrt(kxx - a)
        return var
        
    def calc_alpha(self, K, noise_var, y):
        alpha = np.linalg.inv((K + noise_var * np.eye(self.N))).dot(y)
        return alpha
    
    def calc_gamma(self, K, noise_var, X):
        beta = self.get_Beta(X)
        gamma = np.multiply(np.linalg.inv((K + noise_var * np.eye(self.N))), beta)
        return gamma
    
    def get_Beta(self, X):
        N = X.shape[0]
        Beta = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                Beta[i,j] = self.beta(X[i], X[j])
        return Beta
    
    def beta(self, x1, x2):
        W = self.gpr.W
        beta_x1x2 = np.exp(-0.25 * (x1 - x2).T.dot(W).dot(x1 - x2))
        return beta_x1x2

    def get_q_mean(self, x, X):
        q_mean = 0
        kpp = self.gpr.k_prime_prime
        for i in range(self.N):
            q_mean += np.absolute(kpp(x, X[i])) * self.alpha[i]
        return q_mean

class GPR(GaussianProcessRegressor):
    def __init__(self, X, y, convergence_rate=1.0):
        self.X = X
        self.y = y
        self.W = None
        self.K = None
        self.noise_var = None
        self.converge_rate_sq = np.power(convergence_rate, 2)
        GaussianProcessRegressor.__init__(self, kernel=self.get_kernel(self.X))
        self.fit(self.X, self.y)
        self.update_W()
        self.update_K(self.X)
        self.update_noise_var()
	
    def update(self, X, y):
        self.X = np.vstack((self.X, X))
        self.y = np.append(self.y, y)
        self.fit(self.X, self.y)
        self.update_W()
        self.update_K(self.X)
        self.update_noise_var()
    
    def get_kernel(self, X):
        length_scale = np.random.normal(loc=1.0,scale=.1,size=X.shape[1])
        rbf = kernels.RBF(length_scale=length_scale)
        wk = kernels.WhiteKernel()
        kernel = rbf + wk
        return kernel

    def update_K(self, X):
        N = X.shape[0]
        K = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                K[i,j] = self.k(X[i], X[j])
        self.K = K

    def update_W(self):
        length_scales = self.kernel_.get_params()['k1__length_scale']
        if X.shape[1] == 1:
            length_scales = np.array([length_scales]) # length scales is a singular number, because X is a one dimensional dataset, hence we make it into an array
            
        w = 1.0 / np.power(length_scales, 2)
        W = np.diag(w)
        self.W = W

    def update_noise_var(self):
        noise_var = self.kernel_.get_params()['k2__noise_level']
        self.noise_var = noise_var


    def k(self, x1, x2):
        k_x1x2 = self.converge_rate_sq * np.exp(-0.5 * (x1 - x2).T.dot(self.W).dot(x1 - x2))
        return k_x1x2

    def k_prime(self, x1, x2):
        k_prime_x1x2 = self.converge_rate_sq * np.exp(-1.0 * (x1 - x2).T.dot(self.W).dot(x1 - x2))
        return k_prime_x1x2
             
    def k_prime_prime(self, x1, x2):
        k_prime_prime_x1x2 = self.converge_rate_sq * np.exp(-1.0/6.0 * (x1 - x2).T.dot(self.W).dot(x1 - x2))
        return k_prime_prime_x1x2


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    n = 30
    X = np.atleast_2d(4 * np.random.rand(n)).T
    y = np.cos(X) + 0.1 * np.random.randn(n, 1)
    
    test_x = np.linspace(0, 2 * np.pi)
    gpbandit = ContinuumArmedBandit(X, y)
    
    pred, ci = gpbandit.predict(np.atleast_2d(test_x).T)
    pred = np.squeeze(pred)  # There is only one output dimension.
    ci = np.squeeze(ci)
    
    plt.fill_between(test_x, pred - ci, pred + ci, color=(0.8, 0.8, 1.0))
    plt.plot(test_x, pred)
    plt.scatter(X, y)
    plt.show()

