# 1. Khởi tạo thư viện cần thiết
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from mpl_toolkits.mplot3d import Axes3D

def check_convergence(X, y, w):
    y_pred = np.sign(np.dot(X, w))
    return np.where(y_pred != y)[0]  # trả về 1 tuple chứa các idx của các phần tử khác label


def visualize_data(X, X_concat, y, w, epochs):
    plt.clf()
    X_pos = X[y == 1]
    X_neg = X[y == -1]

    ax = plt.axes(projection='3d')
    ax.scatter3D(X_pos[:, 0], X_pos[:, 1], X_pos[:, 2], marker='x', color='red', label='y==1')
    ax.scatter3D(X_neg[:, 0], X_neg[:, 1], X_neg[:, 2], marker='o', color='blue', label='y==-1')

    x = np.linspace(-5, 5, 30)
    y = np.linspace(-5, 5, 30) 
    x, y = np.meshgrid(x, y)  # tạo lưới 2 chiều
    # Phương trình mặt phẳng: w0*x0 + w1*x1 + w2*x2 + w3*x3 = 0
    if abs(w[3]) < 10e-7:
        w[3] = w[3] + 10e-5  # chống w = 0
    z = (-w[0] - w[1] * x - w[2]*y) / (w[3])

    ax.plot_surface(x, y, z, alpha=0.5, color = 'yellow', cmap='viridis')
    plt.title('Number interations: ' + str(epochs))

    plt.pause(0.15)


def PCA_Algorithm(X, X_concat, y, w):
    epochs = 0
    while True:
        idx_list = check_convergence(X_concat, y, w)
        if len(idx_list) == 0:
            break  # đã convergence

        idx = np.random.choice(idx_list)  # choose random
        # Update
        w = w + X_concat[idx] * y[idx]
        epochs += 1
        visualize_data(X, X_concat, y, w, epochs)
    return w


def main():
    plt.ion()
    X, y = make_classification(n_samples= 30, n_features= 3, n_classes=2, n_redundant=0, random_state = 42)
    y = np.where(y <= 0, -1, 1)  # trả về mảng với phần tử âm thì label = -1, ngược lại thì label = 1
    # (30,)

    X_ones = np.ones((X.shape[0], 1))
    X_concat = np.concatenate((X_ones, X), axis=1)  # mở rộng x thêm phần tử x0 = 1
    # (30,4)

    # Khởi tạo params w có shape bằng shape của X_concat
    w_init = np.random.randn(X_concat.shape[1])  # (4,)
    print('Tham số w khởi tạo ban đầu là: ', w_init)
    w = PCA_Algorithm(X, X_concat, y, w_init)

    print('Tham số sau khi chạy thuật toán là: ', w)

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()




