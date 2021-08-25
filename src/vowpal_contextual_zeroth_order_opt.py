# -*- coding: utf-8 -*-
from vowpalwabbit import pyvw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    vw = pyvw.vw('--cbzo --policy linear -l 0.01 --radius 0.1 --quiet')

    costs_progress = []

    for _ in range(500):
        # Get context from environment
        ctx = Environment.next_ctx()

        # Determine what action to take for this context by calling predict()
        # on an unlabelled example (which includes the context)
        ex = vw.parse(' | c1:{} c2:{}'.format(ctx[0], ctx[1]), labelType=vw.lContinuous)
        pred = vw.predict(ex)
        vw.finish_example(ex)

        # A list (length=2) of pdf segments are returned. Sample
        # action from them
        action, pdf_value = sample_action(pred)

        # Get cost for the action and pass it to VW by creating
        # a labelled example
        cost = Environment.cost(ctx, action)
        ex = vw.parse('ca {}:{}:{} | c1:{} c2:{}'.format(action, cost, pdf_value, ctx[0], ctx[1]), labelType=vw.lContinuous)
        vw.learn(ex)
        vw.finish_example(ex)

        costs_progress.append(cost)

    vw.finish()
    plot(costs_progress)

"""
We create a synthetic environment where the optimal action is linear
and the cost is the absolute loss function
"""
class Environment:
    wopt, bopt = [-3, 5], 0.5
    rs = np.random.RandomState(0)

    @staticmethod
    def next_ctx():
        return Environment.rs.normal(1, 1, size=(2,))

    @staticmethod
    def cost(ctx, action):
        optimal_action = np.dot(Environment.wopt, ctx) + Environment.bopt
        return abs(optimal_action - action)

"""
Samples action from the prediction.
pred is of the form [(left1, right1, pdf_value1), (left2, right2, pdf_value2)]
"""
def sample_action(pred):
    # the next line is equivalent to p = [0.5, 0.5]
    p = np.array([pred[i][2] * (pred[i][1] - pred[i][0]) for i in range(2)])
    idx = np.random.choice(2, p=p / p.sum())

    return np.random.uniform(pred[idx][0], pred[idx][1]), pred[idx][2]

def plot(costs_progress):
    _, ax = plt.subplots()
    costs_progress = pd.Series(costs_progress).rolling(10).mean()
    ax.plot(costs_progress, marker='.', markersize=2, linewidth=1)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost (rolling mean)')
    ax.grid()
    plt.savefig('progress.png')

if __name__ == '__main__':
    np.random.seed(0)
    main()
