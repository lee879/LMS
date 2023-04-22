import numpy as np
import matplotlib.pyplot as plt

np.random.seed(666)
# 生成含有高斯噪声的正弦波形数据
N = 1000  # 数据点数
t = np.linspace(0, 4*np.pi, N)  # 生成时间序列
y = np.sin(t) + 0.1 * np.random.randn(N)  # 生成带有高斯噪声的正弦波形数据
Y = np.sin(t)
# 初始化LMS算法参数
mu = 0.01  # 步长，控制每一步调整参数的程度
order = 10  # 滤波器阶数，也就是模型参数的个数
w = np.zeros(order)  # 初始化模型参数为0

# 应用LMS算法去噪
y_est = np.zeros(N)  # 用于存储LMS算法输出的去噪后的数据
for i in range(order, N):
    # 构建输入向量，即取最近的order个样本点
    x = y[i-order:i]
    # 使用当前的模型参数计算输出值
    y_est[i] = np.dot(w, x)
    # 计算误差
    e = y[i] - y_est[i]
    # 更新模型参数
    w = w + mu * e * x

# 绘制去噪前后的数据对比图
plt.plot(t, y, label='date_noise')
plt.plot(t, y_est, label='date_lms')
plt.plot(t,Y,label="date_standard")
plt.legend()
plt.show()
