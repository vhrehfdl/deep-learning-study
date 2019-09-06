import numpy as np


def mean_squared_error(y, t):
    return (1/2) * np.sum((y-t)**2)


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))


def main():
    # Answer
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    # Predict
    y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    print(mean_squared_error(np.array(y1), np.array(t)))
    print(cross_entropy_error(np.array(y1), np.array(t)))


    # Predict
    y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
    print(mean_squared_error(np.array(y2), np.array(t)))
    print(cross_entropy_error(np.array(y2), np.array(t)))


if __name__ == '__main__':
    main()
