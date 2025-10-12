# 1. Import thư viện cần thiết
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification  # thư viện generate data ngẫu nhiên theo phân phối gaussian




def visualize_data(X, y, w, update_num):
    plt.clf()
    X_pos = X[y == 1]
    X_neg = X[y == -1]

    plt.scatter(X_pos[:, 0], X_pos[:, 1], marker='x', color='red', label='y == 1')

    plt.scatter(X_neg[:, 0], X_neg[:, 1], marker='o', color='blue', label='y == -1')

    x = np.linspace(-3, 5, 50)
    if abs(w[2]) < 10e-7:
        w[2] = w[2] + 10e-5  # chống w = 0
    y = (-w[1] * x - w[0]) / (w[2])

    plt.plot(x, y, color='green', linewidth=1.5)

    plt.title('Number interations: ' + str(update_num))

    plt.pause(0.25)


def check_convergence(X_concat, y, w):  # hàm thực hiện check xem liệu model đã convergence hay chưa
    predict = np.sign(np.dot(X_concat, w))
    return np.where(predict != y)[0]  # trả về 1 tuple các idx mà misclassified


def PLA_Algorithm(X, X_concat, y, w):
    update_num = 0  # biến đếm số lần lặp thực hiện cập nhật thông số
    while True:
        visualize_data(X, y, w, update_num)

        idx_list = check_convergence(X_concat, y, w)  # idx mà các data point bị misclassified
        if len(idx_list) == 0: break  # không có idx trả về hay đã convergence

        # choose randomly misclass point
        idx = np.random.choice(idx_list)

        # update
        w = w + y[idx] * X_concat[idx]

        update_num += 1

    return w


def main():
    # 2. Chuẩn bị dữ liệu
    plt.ion()  # bật chế độ tương tác
    plt.figure()
    X, y = make_classification(n_samples=30, n_features=2, n_redundant=0, n_classes=2, random_state=42)
    y = np.where(y == 0, -1, 1)

    X_one = np.ones(shape=(X.shape[0], 1))

    X_concat = np.concatenate((X_one, X), axis=1)  # (8,3)

    w = np.random.randn(X_concat.shape[1])  # initialize the params  (3,1)

    print('Initialize parameters w = ', w)

    w = PLA_Algorithm(X, X_concat, y, w)

    print('Final w = ', w)

    plt.ioff()
    plt.show()  # giữ figure khi chạy xong


if __name__ == '__main__':
    main()

