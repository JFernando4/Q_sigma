from TileCoderApproximator import TileCoderApproximatorAV
from Q_sigma import QSigma
from MountainCar import MountainCar
from joblib import Parallel, delayed
import pickle
from pylab import arange, zeros, size, float64


def train_agent(environment, fa, sig, alph, n, bet, eps, episodes):
    agent = QSigma(env=environment, function_approximator=fa, alph=alph, sig=sig, n=n, beta=bet, eps=eps)
    agent.train(episodes)
    return agent

if __name__ == '__main__':
    num_agents = 200
    alphas = arange(0.1, 1.1, 0.1)
    sigmas = arange(0, 1.2, 0.2)
    num_episodes = 50
    num_cores = 10
    eps = 0.1
    n = 1
    Environment = MountainCar()
    Func_Approx = TileCoderApproximatorAV()

    beta = 0.95
    for i in range(size(alphas)):
        alpha = alphas[i]
        agents = Parallel(n_jobs=num_cores)(delayed(train_agent)
                                            (Environment, Func_Approx, 1, alpha, n, beta, eps, num_episodes)
                                            for i in range(num_agents))
        pickle.dump(agents,
                    open("/home/jfernan/Q_sigma/Agents/S1.0"+"_A"+str(alpha)+"_N"+str(n)+\
                         "_B"+str(beta)+".p", "wb"))


    for sigma in sigmas:
        for i in range(size(alphas)):
            alpha = alphas[i]
            agents = Parallel(n_jobs=num_cores)(delayed(train_agent)
                                                (Environment, Func_Approx, sigma, alpha, n, 1, eps, num_episodes)
                                                for i in range(num_agents))
            pickle.dump(agents,
                        open("/home/jfernan/Q_sigma/Agents/S"+str(sigma)+"_A"+str(alpha)+"_N"+str(n)+\
                                 "_B1.p", "wb"))
