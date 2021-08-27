import numpy as np
from sklearn.gaussian_process import kernels
from sklearn.gaussian_process import GaussianProcessRegressor

class ContinuumArmedBandit:
    """
    This class implements ContinuumArmedBandit, which is a multi-armed bandit extended
    to work in the continuous action space definition of the problem. The method of achieving this is 
    by keeping track of the distribution of rewards and expected rewards for each point in the action space.
    In the usual MAB setting this boils down to keeping track of summary statistics for each action and using algorithms like
    LinUCB, Exp3 etc. to sample from the space itself.
    
    In our case, we do something similar in that we use the methodology of LinUCB and use the upper confidence interval predicted
    for the rewards around a given action in order to pick the next best action to sample, however, 
    we approximate the probabilities differently in this continuous setting: We use 
    Gaussian processes to approximate the distribution of rewards varying across our continuous action space!
    
    This allows us to use simple to use and easy to extend methods such as GaussianProcessRegressor from scikit-learn (see docs for more).
    
    We merely extend this class in the GPR implementation which accompanies this class, in order to implement this acquisition function approach in 
    order to sample our probability space for the next best action.
    
    """
    def __init__(self, X, y, convergence_rate=1.0):
        """
        Intialize the class with the necessary X and y values, in this case representing the actions and rewards to train on, respectively.
        The convergence rate determines by what factor to update the kernel and Gaussian parameters W, K, alpha and gamma
        
        Parameters
        ----------
        X : array(n_samples, n_actions) or pd.DataFrame of shape (n_samples, n_actions)
            The actions to train on. Usually n_actions is 1, meaning that shape (n_samples, ) results for both input types
        y : array(n_samples, ) or pd.DataFrame or pd.Series of shape (n_samples, 1)
            The rewards which result from the corresponding actions in X above.
            
        Returns
        -------
            
        """
        self.X = X.copy()
        self.y = y.copy() # set X and y as local parameters
        self.gpr = GPR(self.X, self.y, convergence_rate=convergence_rate) # define our custom defined GaussianProcessRegressor model
        self.N = self.X.shape[0]
        self.convergence_rate = convergence_rate
        self.alpha = self.calc_alpha(self.gpr.K, self.gpr.noise_var, self.y) # update the alpha values associated with reward or y-values
        self.gamma = self.calc_gamma(self.gpr.K, self.gpr.noise_var, self.X) # update the gamma values associated with actions or data points or x-values

    def select_action(self):
        """
        Select the next action to sample from. Can be thought of as similar to an acquisition function in Bayesian optimization

        Parameters
        ----------

        Returns
        -------
        x : float
            The result is the numeric action which will act as our next continuous action to take for this instance.
            
        """
        x = self.get_x_best(self.X)
        return x
    
    def update(self, X, y):
        """
        Stacks the new X and y values onto the existing ones, updating the parameters of the model and hence refitting it totally on the new appended data.        

        Parameters
        ----------
        X : array(n_samples, ) 
            An array of points which correspond to the action points to update the model with
            
        y : array(n_samples)
            An array containing the associated reward values to model against X

        Returns
        -------

        """
        self.X = self.X.append(X) # append copies of this dataset in the locally stored class values
        self.y = self.y.append(y) 
        self.gpr = GPR(self.X, self.y, convergence_rate=self.convergence_rate)
        self.N = self.X.shape[0] # update dataset size convenience parameter

        self.alpha = self.calc_alpha(self.gpr.K, self.gpr.noise_var, self.y) # update alpha for this new dataset and post-retraining of the GPR model
        self.gamma = self.calc_gamma(self.gpr.K, self.gpr.noise_var, self.X) # update gamma for this new dataset and post-retraining of the GPR model

    def get_x_best(self, X):
        """
        Selects the next action to take based on the acquisition function defined for the continuum gaussian bandit        

        Parameters
        ----------
        X : array(n_samples, ) 
            An array of points for which to calculate a merit score via the acquisition function of the bandit model
            thus giving rise to candidate points to choose for the next query point x_best
            
        Returns
        -------
        x_best : float
            The next best point to try sample at as the next action
        """
        
        merit_best = 0
        x_best = None
        for j in range(self.N):
            x = X[j]
            s = self.get_s(x, X)
            x = x + s
            x_merit = self.get_merit(x, X)
            if x_merit > merit_best:
                x_best = x
                merit_best = x_merit
        return x_best

    def get_s(self, x, X):
        """
        Returns the mean prediction for the value x relative to the action space X, with confidence envelope s. 
        Works in the derivative space to approximate gradients of the model fit       

        Parameters
        ----------
        x : float type
            the action query point to evaluate the mean and standard deviation derivatives for following from the kernel function
        X : array(n_samples, ) 
            An array of points which correspond to the action points to update the model with
            
        Returns
        -------
        s : float
            The derivative mean prediction for point x relative to X with derivative envelope 
        """
        der_mean = self.get_derivative_mean(x, X)
        der_std = self.get_derivative_std(x, X)
        s = der_mean + der_std
        return s
    
    def get_derivative_std(self, x, X):
        """
        Returns the derivative standard deviation prediction for the point x relative to action space data X w.r.t. the kernel       

        Parameters
        ----------
        x : float type
            the action query point to evaluate the mean and standard deviation derivatives for following from the kernel function
        X : array(n_samples, ) 
            An array of points which correspond to the action points to update the model with
            
        Returns
        -------
        s : float
            The derivative standard deviation envelope for point x relative to X w.r.t. the kernel
        """
        der_std = 0
        W = self.gpr.W
        k_prime = self.gpr.k
        std = self.get_std(x, X)
        for i in range(self.N):
            for j in range(self.N):
                der_std += (2.0 / std) * self.gamma[i,j] * W.dot((X[i] + X[j]) / 2.0 - x) * k_prime(x, (X[i] + X[j]) / 2.0) 
        return der_std


    def get_derivative_mean(self, x, X):
        """
        Retrieves the model kernel for the dataset X provided. TODO: update this method to simply use self.X. 
        At the moment uses a user provided X, which in this case is self.X passed internally among the class methods.
        

        Parameters
        ----------
        X : array(n_samples, ) 
            An array of points which correspond to the action points to update the model with
            
        Returns
        -------
        kernel : sklearn.gaussian_process.kernels types
            A kernel class which consists of the kernel matrix with normalized length scales along each dimension of X, determined by X.shape[1] of course. 
            it is a combination between a radial basis function kernel and some white noise
        """
        der_mean = 0
        W = self.gpr.W
        k = self.gpr.k
        for i in range(self.N):
            der_mean += W.dot(X[i] - x) * k(x, X[i]) * self.alpha[i]
        return der_mean


    def get_merit(self, x, X):
        """
        Retrieves the model kernel for the dataset X provided. TODO: update this method to simply use self.X. 
        At the moment uses a user provided X, which in this case is self.X passed internally among the class methods.
        

        Parameters
        ----------
        X : array(n_samples, ) 
            An array of points which correspond to the action points to update the model with
            
        Returns
        -------
        kernel : sklearn.gaussian_process.kernels types
            A kernel class which consists of the kernel matrix with normalized length scales along each dimension of X, determined by X.shape[1] of course. 
            it is a combination between a radial basis function kernel and some white noise
        """
        mean = self.get_mean(x, X)
        var = self.get_std(x, X)
        merit = mean + var
        return merit
    
    def predict(self):
        """
        Retrieves the model kernel for the dataset X provided. TODO: update this method to simply use self.X. 
        At the moment uses a user provided X, which in this case is self.X passed internally among the class methods.
        

        Parameters
        ----------
        X : array(n_samples, ) 
            An array of points which correspond to the action points to update the model with
            
        Returns
        -------
        kernel : sklearn.gaussian_process.kernels types
            A kernel class which consists of the kernel matrix with normalized length scales along each dimension of X, determined by X.shape[1] of course. 
            it is a combination between a radial basis function kernel and some white noise
        """
        test_x = np.linspace(0.01, np.max(self.X))
        
        val_pred, val_var = self.predict_reward(np.atleast_2d(test_x).T)
        
        
        action_idx = np.where(val_pred + val_var == np.max(val_pred + val_var))
        action = self.X[action_idx]
        
        return action
        
    
    def predict_reward(self, x_arr):
        """
        Retrieves the model kernel for the dataset X provided. TODO: update this method to simply use self.X. 
        At the moment uses a user provided X, which in this case is self.X passed internally among the class methods.
        

        Parameters
        ----------
        X : array(n_samples, ) 
            An array of points which correspond to the action points to update the model with
            
        Returns
        -------
        kernel : sklearn.gaussian_process.kernels types
            A kernel class which consists of the kernel matrix with normalized length scales along each dimension of X, determined by X.shape[1] of course. 
            it is a combination between a radial basis function kernel and some white noise
        """
        val_pred = []
        val_var = []
        for x in x_arr:
            mean = self.get_mean(x, self.X)
            var = self.get_std(x, self.X)
            val_pred.append(mean)
            val_var.append(var)
        return val_pred, val_var

    def get_mean(self, x, X):
        """
        Retrieves the model kernel for the dataset X provided. TODO: update this method to simply use self.X. 
        At the moment uses a user provided X, which in this case is self.X passed internally among the class methods.
        

        Parameters
        ----------
        X : array(n_samples, ) 
            An array of points which correspond to the action points to update the model with
            
        Returns
        -------
        kernel : sklearn.gaussian_process.kernels types
            A kernel class which consists of the kernel matrix with normalized length scales along each dimension of X, determined by X.shape[1] of course. 
            it is a combination between a radial basis function kernel and some white noise
        """
        mean = 0
        k = self.gpr.k
        for j in range(self.N):
            mean += k(x,X[j]) * self.alpha[j]
        return mean

    def get_std(self, x, X):
        """
        Retrieves the model kernel for the dataset X provided. TODO: update this method to simply use self.X. 
        At the moment uses a user provided X, which in this case is self.X passed internally among the class methods.
        

        Parameters
        ----------
        X : array(n_samples, ) 
            An array of points which correspond to the action points to update the model with
            
        Returns
        -------
        kernel : sklearn.gaussian_process.kernels types
            A kernel class which consists of the kernel matrix with normalized length scales along each dimension of X, determined by X.shape[1] of course. 
            it is a combination between a radial basis function kernel and some white noise
        """
        
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
        """
        Retrieves the model kernel for the dataset X provided. TODO: update this method to simply use self.X. 
        At the moment uses a user provided X, which in this case is self.X passed internally among the class methods.
        

        Parameters
        ----------
        X : array(n_samples, ) 
            An array of points which correspond to the action points to update the model with
            
        Returns
        -------
        kernel : sklearn.gaussian_process.kernels types
            A kernel class which consists of the kernel matrix with normalized length scales along each dimension of X, determined by X.shape[1] of course. 
            it is a combination between a radial basis function kernel and some white noise
        """
        alpha = np.linalg.inv((K + noise_var * np.eye(self.N))).dot(y)
        return alpha
    
    def calc_gamma(self, K, noise_var, X):
        """
        Retrieves the kernel matrix for the dataset X provided. TODO: update this method to simply use self.X. 
        At the moment uses a user provided X, which in this case is self.X passed internally among the class methods.
        

        Parameters
        ----------
        X : array(n_samples, ) 
            An array of points which correspond to the action points to update the model with
            
        Returns
        -------
        kernel : array(n_actions, n_actions) type, n_action = X.shape[0]
            A kernel matrix with shape the first dimension of X, determined by X.shape[0] of course. 
            it is a matrix which applies the kernel function of choice to each point in the data
        """
        beta = self.get_Beta(X)
        gamma = np.multiply(np.linalg.inv((K + noise_var * np.eye(self.N))), beta)
        return gamma
    
    def get_Beta(self, X):
        """
        Retrieves the Beta matrix for the dataset X provided. TODO: update this method to simply use self.X. 
        At the moment uses a user provided X, which in this case is self.X passed internally among the class methods.
        

        Parameters
        ----------
        X : array(n_samples, ) 
            An array of points which correspond to the action points to update the model with
            
        Returns
        -------
        Beta : array(n_actions, n_actions) type, n_action = X.shape[0]
            A beta matrix with shape the first dimension of X, determined by X.shape[0] of course. 
            it is a matrix which applies the beta function of choice to each point in the data
        """
        N = X.shape[0]
        Beta = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                Beta[i,j] = self.beta(X[i], X[j])
        return Beta
    
    def beta(self, x1, x2):
        """
        Retrieves the beta coefficient between x1 and x2, which is an unscaled kernel function between them used in calculating self.beta

        Parameters
        ----------
        x1 : float 
            A specific point to calculate beta_x1x2  
        x2 : float 
            A specific point to calculate beta_x1x2
            
        Returns
        -------
        beta_x1x2 : float
            The average  beta for x1, x2
        """
        W = self.gpr.W
        beta_x1x2 = np.exp(-0.25 * (x1 - x2).T.dot(W).dot(x1 - x2))
        return beta_x1x2

    def get_q_mean(self, x, X):
        """
        Calculates the mean q value or q parameter of x given data X, using the second kernel function derivative.
        
        This can be interpreted as the average change in the slope relative to other data points
        
        Parameters
        ----------
        x : float
            A specific point to calculate q at relative to X
        X : array(n_samples, ) 
            An array of points which correspond to the action points to update the model with
            
        Returns
        -------
        q_mean : float
            The average value of q for x given X
        """
        q_mean = 0
        kpp = self.gpr.k_prime_prime
        for i in range(self.N):
            q_mean += np.absolute(kpp(x, X[i])) * self.alpha[i]
        return q_mean

class GPR(GaussianProcessRegressor):
    def __init__(self, X, y, convergence_rate=1.0):
        """
        Intialize the class with the necessary X and y values, in this case representing the actions and rewards to train on, respectively.
        The convergence rate determines by what factor to update the kernel and Gaussian parameters W, K, alpha and gamma
        
        Parameters
        ----------
        X : array(n_samples, n_actions) or pd.DataFrame of shape (n_samples, n_actions)
            The actions to train on. Usually n_actions is 1, meaning that shape (n_samples, ) results for both input types
        y : array(n_samples, ) or pd.DataFrame or pd.Series of shape (n_samples, 1)
            The rewards which result from the corresponding actions in X above.
            
        Returns
        -------
            
        """
        
        self.X = X # set the data set and initialize some class variables
        self.y = y
        self.W = None
        self.K = None
        self.noise_var = None
        self.converge_rate_sq = np.power(convergence_rate, 2) # set the square of the convergence for the super model
        GaussianProcessRegressor.__init__(self, kernel=self.get_kernel(self.X)) # get the kernel corresponding to this dataset
        self.fit(self.X, self.y) # fit the model using the super class fit method
        self.update_W() # update the W matrix with length scale parameters
        self.update_K(self.X) # update the kernel matrix using the kernel and the dataset
        self.update_noise_var() # update the noise variable using the super class noise variable definition
	
    def update(self, X, y):
        """
        Stacks the new X and y values onto the existing ones, updating the parameters of the model and hence refitting it totally on the new appended data.        

        Parameters
        ----------
        X : array(n_samples, ) 
            An array of points which correspond to the action points to update the model with
            
        y : array(n_samples)
            An array containing the associated reward values to model against X

        Returns
        -------

        """
        
        self.X = np.vstack((self.X, X.copy()))
        self.y = np.append(self.y, y.copy())
        self.fit(self.X, self.y)
        self.update_W()
        self.update_K(self.X)
        self.update_noise_var()
    
    def get_kernel(self, X):
        """
        Retrieves the model kernel for the dataset X provided. TODO: update this method to simply use self.X. 
        At the moment uses a user provided X, which in this case is self.X passed internally among the class methods.
        
        Notes
        -----
        
        TODO: Adjust the kernel to be problem specific. This is very hard to do generally, as it requires domain specific knowledge in ranges of values
        your features will take. See here for more explanation. 
        
        SEE HERE: https://github.com/scikit-learn/scikit-learn/issues/7563
        
        Parameters
        ----------
        X : array(n_samples, ) 
            An array of points which correspond to the action points to update the model with
            
        Returns
        -------
        kernel : sklearn.gaussian_process.kernels types
            A kernel class which consists of the kernel matrix with normalized length scales along each dimension of X, determined by X.shape[1] of course. 
            it is a combination between a radial basis function kernel and some white noise
        """
        length_scale = np.random.normal(loc=1.0,scale=.1,size=X.shape[1])
        rbf = kernels.RBF(length_scale=length_scale)
        # we set these particular levels of noise to ensure better convergence of the GPR (tested in practice to work better)
        wk = kernels.WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-4))
        kernel = rbf + wk
        return kernel

    def update_K(self, X):
        """
        Retrieves the kernel matrix for the dataset X provided. TODO: update this method to simply use self.X. 
        At the moment uses a user provided X, which in this case is self.X passed internally among the class methods.
        

        Parameters
        ----------
        X : array(n_samples, ) 
            An array of points which correspond to the action points to update the model with
            
        Returns
        -------
        kernel : array(n_actions, n_actions) type, n_action = X.shape[0]
            A kernel matrix with shape the first dimension of X, determined by X.shape[0] of course. 
            it is a matrix which applies the kernel function of choice to each point in the data
        """
        N = X.shape[0]
        K = np.zeros((N,N)) # generate empty N x N matrix of zeroes to instantiate
        for i in range(N):
            for j in range(N):
                K[i,j] = self.k(X[i], X[j]) # apply kernel function of choice defined below to each pair of data points
        self.K = K # set the kernel in the class variables

    def update_W(self):
        """
        Retrieves the kernel matrix for the dataset X provided. TODO: update this method to simply use self.X. 
        At the moment uses a user provided X, which in this case is self.X passed internally among the class methods.
        

        Parameters
        ----------
        X : array(n_samples, ) 
            An array of points which correspond to the action points to update the model with
            
        Returns
        -------
        kernel : sklearn.gaussian_process.kernels types
            A kernel class which consists of the kernel matrix with normalized length scales along each dimension of X, determined by X.shape[1] of course. 
            it is a combination between a radial basis function kernel and some white noise
        """
        length_scales = self.kernel_.get_params()['k1__length_scale']
        if len(self.X.shape) == 1 or self.X.shape[1] == 1:
            length_scales = np.array([length_scales]) # length scales is a singular number, because X is a one dimensional dataset, hence we make it into an array
            
        w = 1.0 / np.power(length_scales, 2)
        W = np.diag(w)
        self.W = W

    def update_noise_var(self):
        """
        Updates the noise variable in the GaussianProcessRegressor model and internally based on updated values
        of self.X and self.y

        Parameters
        ----------

        Returns
        -------
        """
        noise_var = self.kernel_.get_params()['k2__noise_level']
        self.noise_var = noise_var


    def k(self, x1, x2):
        """
        Kernel function defined as variant of matern kernel. Takes in two numeric vectors, x1 and x2 from
        the data self.X and applies the kernel to them. Returns kernel value to be used in 
        constructing a kernel matrix

        Parameters
        ----------
        x1 : float or array(1, n_dim) if multi-dimensional.
            The first data point to be applied against the kernel with x2
        x2 : float or array(1, n_dim) if multi-dimensional.
            The second data point to be applied against the kernel with x1

        Returns
        -------
        k_x2x2 : float type
            The return value which is the result of the kernel function applied to x1 and x2
            
        """
        k_x1x2 = self.converge_rate_sq * np.exp(-0.5 * (x1 - x2).T.dot(self.W).dot(x1 - x2))
        return k_x1x2

    def k_prime(self, x1, x2):
        """
        Kernel first derivative of the usual kernel function defined in self.k
        This is a helper function for calculating gradients, increases, decreases etc.
        When arriving at query points via the acquisition functions


        Parameters
        ----------
        x1 : float or array(1, n_dim) if multi-dimensional.
            The first data point to be applied against the kernel with x2
        x2 : float or array(1, n_dim) if multi-dimensional.
            The second data point to be applied against the kernel with x1

        Returns
        -------
        k_x2x2 : float type
            The return value which is the result of the kernel function applied to x1 and x2
            
        """
        k_prime_x1x2 = self.converge_rate_sq * np.exp(-1.0 * (x1 - x2).T.dot(self.W).dot(x1 - x2))
        return k_prime_x1x2
             
    def k_prime_prime(self, x1, x2):
        """
        Kernel second derivative of the usual kernel function defined in self.k
        This is a helper function for calculating gradients, increases, decreases etc.
        When arriving at query points via the acquisition functions

        Parameters
        ----------
        x1 : float or array(1, n_dim) if multi-dimensional.
            The first data point to be applied against the kernel with x2
        x2 : float or array(1, n_dim) if multi-dimensional.
            The second data point to be applied against the kernel with x1

        Returns
        -------
        k_x2x2 : float type
            The return value which is the result of the kernel function applied to x1 and x2
            
        """
        k_prime_prime_x1x2 = self.converge_rate_sq * np.exp(-1.0/6.0 * (x1 - x2).T.dot(self.W).dot(x1 - x2))
        return k_prime_prime_x1x2


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # generate some random data here and there, one dimensional X action points
    # and one dimensional rewards for 2d plotting
    n = 100
    X2 = np.atleast_2d(4 * np.random.rand(n)).T
    y = np.cos(X2) + 0.1 * np.random.randn(n, 1)
    
    # test space for the bandit, we evaluate the merit or the possible reward for each point in x
    
    test_x = np.linspace(0, 2 * np.pi)
    # run the bandit
    gpbandit = ContinuumArmedBandit(X2, y)
    # get predictions in pred and confidence intervals in ci
    pred, ci = gpbandit.predict_reward(np.atleast_2d(test_x).T)
    pred = np.squeeze(pred)  # There is only one output dimension.
    ci = np.squeeze(ci) # expand the output dimension here too
    
    # plot the results, showing a filled area for the confidence intervals
    plt.fill_between(test_x, pred - ci, pred + ci, color=(0.8, 0.8, 1.0))
    plt.plot(test_x, pred)
    plt.scatter(X2, y)
    plt.show()

