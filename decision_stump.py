import numpy as np

def compute_denoised_E_out(s, theta):
    # this function compute E_out(g, 0)
    if s == 1:
        return np.abs(theta) / 2
    elif s == -1:
        return (2 - np.abs(theta)) / 2
    else:
        raise ValueError('the value of s should be -1 or 1')

def compute_noised_E_out(E_out_denoised, tau):
    # this function compute E_out(g, tau)
    return E_out_denoised*(1 - 2*tau) + tau

def evaluate_answer(s, theta, E_in, tau):
    # this function compute E_out(g, tau) - E_in(g)
    denoised_E_out = compute_denoised_E_out(s, theta)
    noised_E_out = compute_noised_E_out(denoised_E_out, tau)
    return noised_E_out - E_in

def generate_data(size, flip_rate):
    
    X = np.random.uniform(-1, 1, size)
    Y_true = np.sign(X)
    flip_decider = np.sign(np.random.rand(size) - flip_rate) < 0
    flip_array = np.where(flip_decider, -1, 1)
    Y = Y_true * flip_array

    return X, Y

class decision_stump():
    def __init__(self):
        self.s = -1
        self.theta = -1

    def predict(self, x):
        return self.s * np.where(x > self.theta, 1, -1)
    
    def fit(self, X, Y):
        #preprocessing of train data
        if len(X) != len(Y):
            raise ValueError("The size of X and Y should be the same!")
        data_size = len(X)
        X_train = np.array(X)
        Y_train = np.array(Y)
        # sort the training data by the value of x
        sorted_index =  X_train.argsort()
        X_train = X_train[sorted_index]
        Y_train = Y_train[sorted_index]

        # building of the table for DP
        prefix_positvie_counter = np.zeros(data_size) # number of +1 in y[:i+1]
        prefix_positvie_counter[0] = 1 if Y_train[0] == 1 else 0
        for i in range(1,data_size):
            if Y_train[i] == 1:
                prefix_positvie_counter[i] = prefix_positvie_counter[i-1] + 1
            else:
                prefix_positvie_counter[i] = prefix_positvie_counter[i-1]
        # case for (s, theta) = (-1, -1)
        best_s = -1
        best_theta = -1
        best_score = data_size - prefix_positvie_counter[-1]
        # case for (s, theta) = (+1, -1)
        if best_score < prefix_positvie_counter[-1]:
            best_s, best_theta = 1, -1
            best_score = prefix_positvie_counter[-1]
        # case for theta = (x_i + x_i+1)/2
        for i in range(0, data_size - 1):
            for s in {-1, 1}:
                # compute the score of positive ray
                score = ((i + 1) - prefix_positvie_counter[i]) \
                        + (prefix_positvie_counter[-1]) \
                        - prefix_positvie_counter[i]
                cur_s, cur_theta = s, (X_train[i] + X_train[i+1])/2
                if s == -1:
                    score = data_size - score # flip the result
                if score > best_score:
                    best_s, best_theta = cur_s, cur_theta
                    best_score = score
                elif score == best_score:
                    if best_s+best_theta > cur_s+cur_theta:
                        best_s, best_theta = cur_s, cur_theta
        self.s, self.theta = best_s, best_theta
        E_in = (data_size - best_score) / data_size
        return self.s, self.theta, E_in

def stump_experiment(exp_times, tau, data_size):
    stump = decision_stump()
    exp_results = []
    for _ in range(exp_times):
        X_train, Y_train = generate_data(data_size, tau)
        s, theta, E_in = stump.fit(X_train, Y_train)
        result = evaluate_answer(s, theta, E_in, tau)
        exp_results.append(result)
    return exp_results
