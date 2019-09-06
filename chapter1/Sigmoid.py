import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    print(sigmoid(10))
    print(sigmoid(100))
    print(sigmoid(0))


if __name__ == '__main__':
    main()
