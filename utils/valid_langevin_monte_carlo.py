# 验证Langevin Monte Carlo是否在步长接近0步数接近无穷时得到真实的分布

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import gaussian_kde

#标准正态分布的表达式为 y = 1/(sqrt(2 * pi)) * e^(-0.5 * x^2)
#log(y) = log(1/(sqrt(2 * pi))) - 0.5 * x^2
#▽log(y) = -x
#根据Langevin Monte Carlo公式有
# x_t+1 = x_t + η * ▽log(y) + sqrt(2*η) * N，其中 N ~ (0, 1)
# x_t+1 = (1 - η) * x_t + sqrt(2*η) * N，其中 N ~ (0, 1)

def plot_probability_density(samples, save_path=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde, norm
    
    # 计算核密度估计
    kde = gaussian_kde(samples)
    x_vals = np.linspace(min(samples), max(samples), 1000)
    kde_vals = kde.evaluate(x_vals)

    # 计算标准正态分布
    normal_vals = norm.pdf(x_vals, 0, 1)  # 均值0，标准差1

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制核密度估计曲线（保留原有蓝色）
    ax.plot(x_vals, kde_vals, 'b-', label='Empirical Density')
    
    # 叠加标准正态分布曲线（中灰色）
    ax.plot(x_vals, normal_vals, color='#666666', linestyle='-', label='Standard Normal')

    ax.set_title('Probability Density Comparison')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def langevin_monte_carlo(x, lr):
    return (1 - lr) * x  + np.sqrt(2 * lr) * np.random.randn()
    #return (1 - lr) * x  #如果不添加随机项，只使用梯度下降项的话，得到的分布和标准正态分布差别较大


if __name__ == "__main__":
    lr = 1e-2
    iter_steps = 5000  #迭代的步数越多，得到的分布越接近标准正态分布
    samples_times = 1000
    sample_value = []
    img_save_path = "./langevin_monte_carlo.png"
    for start_point in tqdm(np.linspace(-10, 10, samples_times)):
        sample_value.append(start_point)
        x = start_point
        for batch in range(iter_steps): 
            sample_value.append(x)
            x = langevin_monte_carlo(x, lr)
    plot_probability_density(sample_value, img_save_path)
        
        
    