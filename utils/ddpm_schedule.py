import numpy as np

alpha_t_list = np.array([np.sqrt(1 - 0.02 * (i+1)/1000) for i in range(1000)])
alpha_t_bar_list = np.cumprod(alpha_t_list)
beta_t_list = np.sqrt(1-alpha_t_list ** 2)
beta_t_bar_list = np.sqrt(1 - alpha_t_bar_list ** 2)