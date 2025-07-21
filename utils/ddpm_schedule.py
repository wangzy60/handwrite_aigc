import numpy as np
import torch

alpha_t_list = np.array([np.sqrt(1 - 0.02 * (i+1)/1000) for i in range(1000)])
alpha_t_bar_list = np.cumprod(alpha_t_list)
beta_t_list = np.sqrt(1-alpha_t_list ** 2)
beta_t_bar_list = np.sqrt(1 - alpha_t_bar_list ** 2)


def add_noise(x, t, shift_func, noise_func, verbose=False):
    if isinstance(shift_func, np.ndarray) and isinstance(noise_func, np.ndarray):
        shift_value = shift_func[t]
        noise_value = noise_func[t]
    else:
        shift_value = shift_func(t)
        noise_value = noise_func(t)
        
    noise = torch.randn_like(x)
    noise_img = noise_value*noise
    img_with_noisy = shift_value*x + noise_img
    return img_with_noisy, noise_img, noise