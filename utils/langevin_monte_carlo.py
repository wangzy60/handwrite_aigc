import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def langevin_monte_carlo_validation(weights, means, sigmas, num_iterations=10000, num_samples=1000, step_size=0.1, save_path='langevin_validation.png'):
    """
    验证郎之万蒙特卡洛方法在迭代后是否收敛到原始混合高斯分布
    
    参数:
    weights: 混合高斯分布的权重列表
    means: 混合高斯分布的均值列表
    sigmas: 混合高斯分布的标准差列表
    num_iterations: 郎之万蒙特卡洛的迭代次数
    num_samples: 采样的样本数量
    step_size: 郎之万蒙特卡洛的步长
    save_path: 保存图像的路径
    """
    
    # 1. 定义混合高斯分布的概率密度函数(PDF)和其对数的梯度
    def mixture_gaussian_pdf(x):
        pdf_val = 0
        for w, m, s in zip(weights, means, sigmas):
            pdf_val += w * norm.pdf(x, m, s)
        return pdf_val
    
    def log_pdf_gradient(x):
        # 计算对数概率密度的梯度
        numerator = 0
        denominator = 0
        for w, m, s in zip(weights, means, sigmas):
            gaussian_val = w * norm.pdf(x, m, s)
            numerator += gaussian_val * (m - x) / (s**2)
            denominator += gaussian_val
        return numerator / denominator if denominator.all() != 0 else 0
    
    # 2. 实现郎之万蒙特卡洛采样
    def langevin_monte_carlo(initial_samples, n_iter, step_size):
        samples = initial_samples.copy()
        for _ in range(n_iter):
            # 郎之万更新规则: x_{t+1} = x_t + ε * ∇log p(x_t) + √(2ε) * z_t
            noise = np.random.normal(0, 1, len(samples))
            gradient = log_pdf_gradient(samples)
            samples = samples + step_size * gradient + np.sqrt(2 * step_size) * noise
        return samples
    
    # 3. 初始化样本
    initial_samples = np.random.normal(-10, 10, num_samples)
    
    # 4. 运行郎之万蒙特卡洛
    final_samples = langevin_monte_carlo(initial_samples, num_iterations, step_size)
    
    # 5. 绘制结果
    plt.figure(figsize=(12, 6))
    
    # 绘制郎之万蒙特卡洛采样结果的直方图
    plt.hist(final_samples, bins=100, density=True, alpha=0.6, color='blue', label='Langevin MC Samples')
    
    # 绘制真实的混合高斯分布
    x = np.linspace(min(means) - 4*max(sigmas), max(means) + 4*max(sigmas), 1000)
    y = mixture_gaussian_pdf(x)
    plt.plot(x, y, 'r-', linewidth=2, label='True Mixture Gaussian')
    
    # 添加图表信息
    plt.title('Langevin Monte Carlo Validation\nAfter {} Iterations'.format(num_iterations))
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return final_samples

# 使用示例
if __name__ == "__main__":
    # 定义混合高斯分布的参数
    weights = [0.3, 0.4, 0.3]  # 权重
    means = [-3, 0, 4]         # 均值
    sigmas = [1, 1.5, 0.8]     # 标准差
    
    # 运行验证
    samples = langevin_monte_carlo_validation(
        weights, means, sigmas, 
        num_iterations=10000, 
        num_samples=5000, 
        step_size=0.1,
        save_path='langevin_validation_result.png'
    )