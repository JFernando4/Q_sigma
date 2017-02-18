from TileCoderApproximator import TileCoderApproximatorAV
from Q_sigma import QSigma
from MountainCar import MountainCar
from joblib import Parallel, delayed
import pickle
from pylab import size


def train_agent(environment, fa, sig, alph, n, bet, eps, episodes):
    agent = QSigma(env=environment, function_approximator=fa, alph=alph, sig=sig, n=n, beta=bet, eps=eps)
    agent.train(episodes)
    return agent

if __name__ == '__main__':
    num_agents = 100
    alphas = [1/10, 1/9, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2, 1]
    alpha_labels = ['10', '9', '8', '7', '6', '5', '4', '3', '2', '1']
    sigmas = [0, 0.25, 0.50, 0.75, 1]
    num_episodes = 100
    num_cores = 4
    eps = 0.1
    n = 4
    Environment = MountainCar()

    beta = 0.95
    for i in range(size(alphas)):
        alpha = alphas[i]
        alpha_label = alpha_labels[i]
        agents = Parallel(n_jobs=num_cores)(delayed(train_agent)
                                            (Environment, TileCoderApproximatorAV(), 1, alpha, n, beta, eps, num_episodes)
                                            for i in range(num_agents))
        pickle.dump(agents,
                    open("/home/jfernando/PycharmProjects/Q_sigma/S1.0"+"_A"+alpha_label+"_N"+str(n) + "_B"+str(beta)+\
                         ".p", "wb"))

    for sigma in sigmas:
        for i in range(size(alphas)):
            alpha = alphas[i]
            alpha_label = alpha_labels[i]
            agents = Parallel(n_jobs=num_cores)(delayed(train_agent)
                                             (Environment, TileCoderApproximatorAV(), sigma, alpha, n, 1, eps, num_episodes)
                                             for i in range(num_agents))
            pickle.dump(agents,
                     open("/home/jfernando/PycharmProjects/Q_sigma/S"+str(sigma)+"_A"+alpha_label+"_N"+str(n) + \
                         "_B1.p", "wb"))
