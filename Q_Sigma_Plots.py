import pickle
import matplotlib.pyplot as plt
from pylab import size, vstack, array, log, sort, inf, zeros, asarray, str_, argmax
from os import listdir


def read_letter_until_character(input_string, initial_index, character):
    """ Reads and store each letter of a word until encounter an specified character """
    mystring = ''
    for i in range(initial_index, len(input_string)):
        letter = input_string[i:(i+len(character))]
        if letter != character:
            mystring += input_string[i]
        else:
            break
    return mystring


def create_table_for_average_return(agents_directory):
    myFiles = listdir(agents_directory)
    arr = None
    sigma = 0
    alpha = 0
    n = 0
    beta = 0
    for onefile in myFiles:
        for k in range(len(onefile)):
            if onefile[k] == 'S':
                sigma = round(float(read_letter_until_character(onefile, k+1, '_')), 2)
            elif onefile[k] == 'A':
                alpha = int(read_letter_until_character(onefile, k+1, '_'))
            elif onefile[k] == 'N':
                n = round(int(read_letter_until_character(onefile, k+1, '_')), 1)
            elif onefile[k] == 'B':
                beta = round(float(read_letter_until_character(onefile, k+1, '.p')), 2)

        agents = pickle.load(open(agents_directory+onefile, 'rb'))
        average = 0
        for agent in agents:
            average += (sum(agent.return_per_episode) / agent.episode_number) / size(agents)

        dtype = [('File_Name', str_, 20), ('n', int), ('sigma', float), ('beta', float), ('alpha', int),
                 ('Average_Return', float)]
        if arr is None:
            arr = array((onefile, n, sigma, beta, alpha, average), dtype=dtype)
        else:
            arr = vstack([arr, array((onefile, n, sigma, beta, alpha, average), dtype=dtype)])
    return arr


def plot_average_return_per_episode(data, n, sigmas=None, betas=None, nlabel=False, lnstyle=None):
    if sigmas is None: sigmas = [0, 0.25, 0.5, 0.75, 1]
    lines = []
    if betas is None: betas = [0.95, 1]
    number_of_lines = 0
    labels = []
    min_return = inf
    for beta in betas:
        for sigma in sigmas:
            temp_lines = [-inf for _ in range(10)]
            temp_data = sort(data[(data['sigma'] == sigma)*(data['n'] == n)*(data['beta'] == beta)], order='alpha')
            if size(temp_data) > 0:
                offset = size(temp_lines) - size(temp_data['Average_Return'])
                for i in range(size(temp_data)):
                    temp_lines[i+offset] = temp_data['Average_Return'][i]
                    if temp_data['Average_Return'][i] < min_return: min_return = temp_data['Average_Return'][i]
                lines.append(temp_lines)
                temp_label = r'$\sigma$ = '+str(sigma)
                if beta != 1: temp_label += r', $\beta$ = '+str(beta)
                if nlabel: temp_label += ", n = "+str(n)
                labels.append(temp_label)
                number_of_lines += 1

    colors = ['b', 'g', 'c', 'k', 'y', 'm', 'b']
    alphas = [i+1 for i in range(10)]
    if lnstyle is None:
        linestyle = (['solid'] * 5)
        linestyle.append('dashed')
    else: linestyle = [lnstyle]*5

    for i in range(number_of_lines):
        plt.plot(alphas, lines[i], color=colors[i], linestyle=linestyle[i], label=labels[i])

    plt.legend(loc=4)
    plt.xlabel(r'$1/\alpha$', fontsize=14)
    plt.ylabel('Average Return per Episode', fontsize=14)
    plt.ylim(ymin=min_return + 100, )


def plot_return_per_episode(file_names, agents_dir, logscale=True, nlabel=True, alpha_label=True):
    all_returns = []
    labels =[]
    total_episodes = []
    for onefile in file_names:
        agents = pickle.load(open(agents_dir+onefile, 'rb'))
        number_of_episodes = agents[0].episode_number
        total_episodes.append([i+1 for i in range(number_of_episodes)])
        temp_returns = zeros(number_of_episodes)
        for agent in agents:
            if logscale: temp_returns += log(-asarray(agent.return_per_episode))
            else: temp_returns += -asarray(agent.return_per_episode)
        all_returns.append(temp_returns/size(agents))

        sigma = 0; beta = 0; alpha = 0; n = 0;

        for k in range(len(onefile)):
            if onefile[k] == 'S':
                sigma = round(float(read_letter_until_character(onefile, k+1, '_')), 2)
            elif onefile[k] == 'A':
                alpha = int(read_letter_until_character(onefile, k+1, '_'))
            elif onefile[k] == 'N':
                n = round(int(read_letter_until_character(onefile, k+1, '_')), 1)
            elif onefile[k] == 'B':
                beta = round(float(read_letter_until_character(onefile, k+1, '.p')), 2)

        temp_label = r'$\sigma$ = '+str(sigma)
        if beta != 1: temp_label += r', $\beta$ = '+str(beta)
        if alpha_label: temp_label += r', $\alpha$ = '+str(alpha)
        if nlabel: temp_label += ', n = '+str(n)

        labels.append(temp_label)

    colors = ['b', 'g', 'c', 'k', 'y', 'm', 'b']
    for i in range(size(file_names)):
        plt.plot(total_episodes[i], all_returns[i], color=colors[i], label=labels[i])

    plt.legend(loc=1)
    plt.xlabel('Episode Number', fontsize=14)
    plt.ylabel('Average Return per Episode', fontsize=14)


def plot_running_average(file_names, agents_dir, window=10, nlabel=True, alpha_label=True):
    running_average = []
    labels =[]
    total_episodes = []
    for onefile in file_names:
        agents = pickle.load(open(agents_dir+onefile, 'rb'))
        number_of_episodes = agents[0].episode_number
        temp_returns = zeros(number_of_episodes)
        for agent in agents:
            temp_returns += -asarray(agent.return_per_episode)
        temp_returns /= size(agents)
        temp_running_average = []
        for i in range(int(number_of_episodes/window)):
            temp_running_average.append(sum(temp_returns[0:((i+1)*window)])/((i+1)*window))
        total_episodes.append([(i+1)*window for i in range(int(number_of_episodes/window))])
        running_average.append(temp_running_average)

        sigma = 0; beta = 0; alpha = 0; n = 0;

        for k in range(len(onefile)):
            if onefile[k] == 'S':
                sigma = round(float(read_letter_until_character(onefile, k+1, '_')), 2)
            elif onefile[k] == 'A':
                alpha = int(read_letter_until_character(onefile, k+1, '_'))
            elif onefile[k] == 'N':
                n = round(int(read_letter_until_character(onefile, k+1, '_')), 1)
            elif onefile[k] == 'B':
                beta = round(float(read_letter_until_character(onefile, k+1, '.p')), 2)

        temp_label = r'$\sigma$ = '+str(sigma)
        if beta != 1: temp_label += r', $\beta$ = '+str(beta)
        if alpha_label: temp_label += r', $\alpha$ = '+str(alpha)
        if nlabel: temp_label += ', n = '+str(n)

        labels.append(temp_label)

    colors = ['b', 'g', 'c', 'k', 'y', 'm', 'b']
    for i in range(size(file_names)):
        plt.plot(total_episodes[i], running_average[i], color=colors[i], label=labels[i])

    plt.legend(loc=1)
    plt.xlabel('Episode Number', fontsize=14)
    plt.ylabel('Average Return at Episode Number', fontsize=14)




def get_maximum(data, sigmas=None, betas=None):
    if sigmas is None: sigmas = [0, 0.25, 0.5, 0.75, 1]
    if betas is None: betas = [0.95, 1]

    file_names = []
    for beta in betas:
        for sigma in sigmas:
            temp_data = data[ (data['sigma']==sigma)*(data['beta']==beta) ]
            if size(temp_data) != 0:
                max_indx = argmax(temp_data['Average_Return'])
                file_names.append(temp_data['File_Name'][max_indx])

    return file_names



