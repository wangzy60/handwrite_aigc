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