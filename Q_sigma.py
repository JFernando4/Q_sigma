from pylab import zeros, argmax, size, unique, inf, uniform, float64, randint


class QSigma:

    def __init__(self, env, function_approximator, eps=0.1, gam=1, alph=0.1, sig=1, n=1, beta=1, initial_theta=None):
        self.epsilon = eps
        self.gamma = gam
        self.sigma = sig
        self.n = n
        self.beta = beta
        # Environment
        self.env = env
        # Function approximator
        self.function_approximator = function_approximator
        self.alpha = alph * self.function_approximator.get_alpha_factor()
        # This is the probability of taking the optimal action
        self.optimal_p = (1-self.epsilon) + (self.epsilon / self.env.get_num_actions())
        # Episode Number: this is the number of training episodes that the agent has received so far
        self.episode_number = 0
        # Return per episode: this is the return per episode that the agent has received
        self.return_per_episode = []
        """ An initial theta can be provided. Otherwise it uses a random initialization """
        if initial_theta is not None:
            self.function_approximator.set_theta(initial_theta)

    """ Setters and Getters for Epsilon, Sigma, n, and Alpha """
    # Epsilon
    def get_epsilon(self):
        return self.epsilon

    def set_epsilon(self, new_epsilon):
        self.epsilon = new_epsilon

    # Sigma
    def get_sigma(self):
        return self.sigma

    def set_sigma(self, new_sigma):
        self.sigma = new_sigma

    # n
    def get_n(self):
        return self.n

    def set_n(self, new_n):
        self.n = new_n

    # Alpha
    def get_alpha(self):
        return self.alpha

    def set_alpha(self, new_alpha):
        self.alpha = new_alpha

    # Episode Number
    def increase_episode_number(self):
        self.episode_number += 1

    # Add Return
    def add_return(self, return_value):
        self.return_per_episode.append(return_value)

    # Here q refers to the action value function. This function returns the approximated
    # action-value function at a given state
    def get_q(self, state):
        q = zeros(self.env.get_num_actions(), dtype=float64)
        for action in range(0, self.env.get_num_actions()):
            q[action] = self.function_approximator.get_value(state, action)
        return q

    # This function returns an action according to an epsilon greedy policy
    def epsilon_greedy_action(self, state):
        p = uniform()
        if p < self.epsilon:
            return randint(0, self.env.get_num_actions())
        else:
            return argmax(self.get_q(state))

    # This function returns the probability of a given action in a given state according to an epsilon greedy policy
    def epsilon_greedy_probability(self, state, action):
        q = self.get_q(state)
        if size(unique(q)) < self.env.get_num_actions():
            max_q = max(q)
            max_observations = 0
            for value in q:
                if value == max_q: max_observations += 1
            probabilities = zeros(size(q))
            for i in range(size(q)):
                if q[i] == max_q: probabilities[i] = ((1-self.epsilon) / max_observations) + \
                                                     (self.epsilon / self.env.get_num_actions())
                else: probabilities[i] = self.epsilon / self.env.get_num_actions()
            return probabilities[action]
        else:
            if action == argmax(q):
                return self.optimal_p
            else:
                return self.epsilon / self.env.get_num_actions()

    # Returns the average over all actions of the action-value function for a given state
    def average_q(self, state):
        q = self.get_q(state)
        average_q = float64(0)
        for action in range(0, self.env.get_num_actions()):
            p = self.optimal_p if action == argmax(q) else (self.epsilon / self.env.get_num_actions())
            average_q += p * q[action]
        return average_q

    # This function trains Q-sigma over a given amount of episodes. Note that train(1) + train(1) = train(2)
    def train(self, num_episodes):
        if num_episodes == 0: return
        rewards_per_episode = zeros(num_episodes)
        Actions = zeros(self.n+1, dtype=int)
        States = [[] for _ in range(self.n + 1)]
        Q = zeros(self.n+1)
        Delta = zeros(self.n)
        Pi = zeros(self.n)
        Sigma = zeros(self.n)

        for episode in range(0,num_episodes):
            self.increase_episode_number()
            S = self.env.init_state()
            A = self.epsilon_greedy_action(S)
            Reward_Sum = 0
            T = inf
            t = 0
            States[t % (self.n+1)] = S
            Actions[t % (self.n+1)] = A
            Q[t % (self.n+1)] = self.function_approximator.get_value(S, A)

            while 1:
                if t < T:
                    R, new_S = self.env.do_action(S, A)
                    States[(t + 1) % (self.n + 1)] = new_S
                    Reward_Sum += R

                    if new_S is None:
                        T = t + 1
                        Delta[t % self.n] = R - self.function_approximator.get_value(States[t % (self.n+1)],
                                                                                     Actions[t % (self.n+1)])
                    else:
                        Sigma[t % self.n] = self.sigma
                        new_A = self.epsilon_greedy_action(new_S)
                        Actions[(t+1) % (self.n+1)] = new_A
                        # Qt = self.function_approximator.get_value(S, A)
                        # Qtplus1 = self.function_approximator.get_value(new_S, new_A)
                        # Delta[t % self.n] = R + (self.gamma * self.sigma * Qtplus1) + \
                        #                     (self.gamma * (1-self.sigma) * self.average_q(new_S)) - Qt
                        Q[(t+1) % (self.n+1)] = self.function_approximator.get_value(new_S, new_A)  #New
                        Delta[t % self.n] = R + (self.gamma * self.sigma * Q[(t+1) % (self.n+1)]) + \
                                            (self.gamma * (1-self.sigma) * self.average_q(new_S)) - Q[t % (self.n+1)] # New

                        Pi[t % self.n] = self.epsilon_greedy_probability(new_S, new_A)
                        S = new_S
                        A = new_A

                Tao = t - self.n + 1
                if Tao >= 0:
                    E = 1
                    # G = self.function_approximator.get_value(States[Tao % (self.n+1)], Actions[Tao % (self.n+1)])
                    G = Q[Tao % (self.n+1)] # New
                    for k in range(Tao, min(Tao + self.n, T)):
                        G += E * Delta[k % self.n]
                        #if (k % self.n) < size(Pi):
                        E = self.gamma * E * ((1-self.sigma) * Pi[k % self.n] + self.sigma)
                    Qtao = self.function_approximator.get_value(States[Tao % (self.n+1)],
                                                                Actions[Tao % (self.n+1)])
                    self.function_approximator.update(States[Tao % (self.n+1)], Actions[Tao % (self.n+1)],
                                                      self.alpha * (G - Qtao))
                t += 1
                if Tao == T - 1: break

            rewards_per_episode[episode] = Reward_Sum
            self.add_return(Reward_Sum)
            self.set_sigma(self.sigma * self.beta)

        return rewards_per_episode

    def test(self, num_episodes):
        if num_episodes == 0: return
        Reward_Sum = 0

        for episode in range(0,num_episodes):
            S = self.env.init_state()
            A = self.epsilon_greedy_action(S)

            while 1:
                R, new_S = self.env.do_action(S, A)
                Reward_Sum += R
                if new_S is None: break
                else:
                    A = self.epsilon_greedy_action(new_S)
                    S = new_S

        return Reward_Sum / num_episodes
