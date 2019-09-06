def NAND(x1, x2):
    if 0.5*x1 + 0.5*x2 >= 0.7:
        return 0
    elif 0.5*x1 + 0.5*x2 < 0.7:
        return 1


def OR(x1, x2):
    if 0.5*x1 + 0.5*x2 <= 0.3:
        return 0
    elif 0.5*x1 + 0.5*x2 > 0.3:
        return 1


def AND(x1, x2):
    if 0.5*x1 + 0.5*x2 <= 0.7:
        return 0
    elif 0.5*x1 + 0.5*x2 > 0.7:
        return 1


def main():
    print("NOT AND Gate 결과")
    print(NAND(0, 0))
    print(NAND(1, 0))
    print(NAND(0, 1))
    print(NAND(1, 1), "\n")

    print("OR Gate 결과")
    print(OR(0, 0))
    print(OR(1, 0))
    print(OR(0, 1))
    print(OR(1, 1), "\n")

    print("AND Gate 결과")
    print(AND(0, 0))
    print(AND(1, 0))
    print(AND(0, 1))
    print(AND(1, 1), "\n")


if __name__ == '__main__':
    main()
