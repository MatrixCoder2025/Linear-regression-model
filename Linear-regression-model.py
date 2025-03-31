# 导入必要的包
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


# 数据加载与标准化
def load_data(file_path, y_column):
    """
    加载数据并进行标准化处理
    :param file_path: 数据文件路径
    :param y_column: 目标列名称
    :return:
        X: 标准化后的特征矩阵
        y: 目标向量
        feature_names: 特征名称列表
    """

    df = pd.read_csv(file_path)

    # 分离特征和目标
    features = df.drop(columns=[y_column])  # 删去因变量那一列

    # 保存统计量
    feature_means = features.mean().values  # 均值向量
    feature_stds = features.std().values   # 标准差向量
    feature_names = features.columns.tolist()  # 提取列表名称
    y = df[y_column].values  # 提取因变量值

    # Z-score标准化
    features = (features - features.mean()) / features.std()

    # 在输入矩阵最左侧插入x0(全为1的列）
    X = np.insert(features.values, 0, 1, axis=1)
    feature_names = ['w_0'] + feature_names

    return X, y, feature_names, feature_means, feature_stds


# 定义线性回归模型
def predict(X, w):
    predictions = np.dot(X, w)
    return predictions


# 定义损失函数（MSE）
def loss(X, w, y):
    m = len(y)
    predictions = predict(X, w)
    return (1 / (2 * m)) * np.sum((predictions - y) ** 2)


# 梯度下降法
def gradient_descent(X, y, w_original, alpha, epochs):
    """
    向量化梯度下降
    :param X: 特征矩阵 (m x (n+1))
    :param y: 目标向量 (m x 1)
    :param w_original: 初始权重向量((n+1) x 1)
    :param alpha: 学习率
    :param epochs: 迭代次数
    :return:
        weights: 优化后的权重
        loss_history: 损失记录
    """

    m = len(y)
    w_new = w_original.copy()
    loss_history = []

    for epoch in range(1, epochs + 1):
        # 计算梯度
        predictions = predict(X, w_new)
        error = predictions - y
        gradients = (1 / m) * np.dot(X.T, error)

        # 更新权重
        w_new -= alpha * gradients

        # 记录损失
        current_loss = loss(X, w_new, y)
        loss_history.append(current_loss)

        # 打印进度
        print(f"Epoch {epoch:4d}/{epochs} | Loss: {current_loss:.4f}")

    return w_new, loss_history


# 将标准化后的权重还原到原始特征空间
def restore_parameters(weights, feature_means, feature_stds):
    w0_prime = weights[0]
    w_prime = weights[1:]

    # 计算原始截距项
    w0 = w0_prime - np.sum((w_prime * feature_means) / feature_stds)

    # 计算原始特征权重
    w = w_prime / feature_stds

    return np.concatenate([[w0], w])


# 主函数
def main():
    file_path = "C:/Users/Administrator/Desktop/pytorchlearning/ex0.csv"   # ！！！此处替换为自己的数据集路径
    y_column = "y"  # ！！！查看你的数据集，找到因变量那一列,填写这列的列名

    # 加载数据
    X, y, feature_names, feature_means, feature_stds = load_data(file_path, y_column)

    # 参数初始化
    np.random.seed()
    w_original = np.random.randn(X.shape[1])
    alpha = 0.001
    epochs = 10000

    # 训练模型
    final_weights, loss_history = gradient_descent(X, y, w_original, alpha, epochs)

    # 输出标准化下训练的结果
    print("标准化下优化后的权重：")
    for name, weight in zip(feature_names, final_weights):
        print(f"{name}: {weight}")

    # 返回参数
    w = restore_parameters(final_weights, feature_means, feature_stds)

    # 输出标准化下训练的结果
    print("返回后的权重：")
    for name, weight in zip(feature_names, w):
        print(f"{name}: {weight}")

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, 'r-', linewidth=5)
    plt.title('Training Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()