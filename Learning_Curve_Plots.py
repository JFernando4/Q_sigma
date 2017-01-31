import matplotlib.pyplot as plt
from pylab import zeros,arange,savefig, size
from TileCoder import numTilings
import pickle

def average_reward_read_data(n=1):
    alphas = arange(0.05, 1.05, 0.05) / numTilings
    sigmas = arange(0, 1.2, 0.2)
    Nstring = "N"+str(n) if n > 1 else ""
    data = []
    for sigma in sigmas:
        sigma_results = []
        for alpha in alphas:
            alpha_data = pickle.load(open("Rewards_Sigma"+str(sigma)+"_Alpha"+str(alpha)+Nstring+".p", 'rb'))
            results = 0
            for agent_data in alpha_data:
                results += sum(agent_data)/size(alpha_data)
            sigma_results.append(results)
        data.append(sigma_results)
    return data


def plot_average_reward(n=1):
    # x should be an n by m matrix or array where n is the number of observations and m is the number of episodes
    data = average_reward_read_data(n)
    alphas = arange(0.05, 1.05, 0.05)
    sigmas = arange(0, 1.2, 0.2)
    colors = ['k', 'b', 'g', 'red', 'c', 'm', 'y']
    handles = []
    for i in range(size(sigmas)):
        plt.plot(alphas, data[i], color=colors[i])
        handles.append(plt.Line2D([], [], color=colors[i], label=r'$\sigma$ ='+str(sigmas[i])))
    plt.legend(handles=handles, loc=4)

def standard_plot(x, y, ylim=None, xlim=None, colors=None, legend=False, legend_labels=None, markers=None, lty=None):
    pass
