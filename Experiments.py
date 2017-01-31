import multiprocessing
import pickle
from joblib import Parallel, delayed
from pylab import arange, size
from MountainCar import MountainCar
from Q_sigma import QSigma
from TileCoderApproximator import TileCoderApproximatorAV


def train_agent(agent, episodes):
    return agent.train(episodes)

if __name__ == '__main__':
    num_agents = 1000
    alphas = arange(0.05, 1.05, 0.05)
    sigmas = arange(0, 1.2, 0.2)
    num_episodes = 500
    num_cores = multiprocessing.cpu_count()
    ns = [1, 2, 4, 8, 16, 32]

    for n in ns:
        for sigma in sigmas:
            for i in range(size(alphas)):
                alpha = alphas[i]
                print("Sigma = ", sigma, "Alpha =", alpha)
                agents = []
                for j in range(num_agents):
                    agents.append(QSigma(env=MountainCar(), function_approximator=TileCoderApproximatorAV(), alph=alpha,
                                         sig=sigma, n=n))
                rewards = Parallel(n_jobs=num_cores)(delayed(train_agent)(agent, num_episodes) for agent in agents)
                pickle.dump(agents,
                            open("Agents_Sigma"+str(sigma)+"_Alpha"+str(alpha)+"N_"+str(n)+".p", "wb"))
