import numpy as np
import pandas as pd
from sklearn.gaussian_process import kernels
from continuum_gaussian_bandits import ContinuumArmedBandit, GPR

class ContextualContinuumArmedBandit:
    def __init__(self, contexts, oracle, bid_max_value, convergence_rate=1.0):
        self.context_dict = {}
        self.contexts = contexts
        self.bid_max_value = bid_max_value
        X = np.arange(0, self.bid_max_value, 100)
        self.num_contexts = len(contexts)
        for context in self.contexts:
            df_context = context.merge(X, how = 'right')
            y_pred = oracle.predict(df_context)
            self.context_dict[context] = (GPR(df_context, y_pred , convergence_rate=convergence_rate), None, None)


    def select_action(self, context):
        contextual_gpr = self.contexts[context][0]
        alpha = self.contexts[context][1]
        gamma = self.contexts[context][2]
        K = contextual_gpr.K
        noise_var = contextual_gpr.noise_var
        alpha = self.calc_alpha(K, noise_var, self.y)
        gamma = self.calc_gamma(K, noise_var, self.X)
        self.contexts[context][1] = alpha
        self.contexts[context][2] = gamma
        x = self.get_x_best(self.X)
        return x
        
    def get_x_best(self, X, context):
        x_best = self.contexts[context][0].get_x_best(X)
        return x_best



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
    
    def fit(self, num_rounds):
        np.random.seed(42)
        for round_num in num_rounds:
            sampled_context = self.contexts[np.random.randint(self.num_contexts)]
            gpr = self.context_dict[sampled_context][0]
            alpha = self.context_dict[sampled_context][1]
            gamma = self.context_dict[sampled_context][2]
            gpr
    
    def predict(self, contexts):
        result_set = []
        for context in contexts:
            try:
                sampled_context = self.context_dict[context]
            except:
                print("Unsampled or unseen context, please update or re-fit the model with the context included.")
                return Exception()
            result_set.append(self.get_x_best(np.arange(0, self.bid_max_value, 100), context))
            
        return np.array(result_set)
    
    def partial_fit(self, contexts_new, X_new, y_new, reward_new):
        for context in contexts_new:
            try:
                sampled_context = self.context_dict[context]
            except:
                print("Unsampled or unseen context, please update or re-fit the model with the context included.")
                return Exception()
            
            
            
            
            

