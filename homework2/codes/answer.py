import decision_stump
import numpy as np

# 15 ~ 18
exp_times_arr = [10000 for _ in range(3)]
tau_arr = [0, 0, 0.1]
data_size_arr = [2, 20, 2]
for exp_times, tau, data_size in \
    zip(exp_times_arr, tau_arr, data_size_arr):
    results = decision_stump.stump_experiment(exp_times, tau, data_size)
    print('Given (exp_times, tau, data_size) = ', (exp_times, tau, data_size))
    print('The average E_out(g, tau) - E_in(g) = ', np.mean(results), '\n')

# 19 ~ 20
exp_times_arr = [10000 for _ in range(2)]
tau_arr = [0.1, 0.1]
data_size_arr = [20, 200]
for exp_times, tau, data_size in \
    zip(exp_times_arr, tau_arr, data_size_arr):
    results = decision_stump.stump_experiment(exp_times, tau, data_size)
    print('Given (exp_times, tau, data_size) = ', (exp_times, tau, data_size))
    print('The median of E_out(g, tau) - E_in(g) = ', np.median(results), '\n')