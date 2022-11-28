import numpy as np
import sympy as sp


class LogColor:
    CEND = '\33[0m'
    CBOLD = '\33[1m'
    CITALIC = '\33[3m'
    CURL = '\33[4m'
    CBLINK = '\33[5m'
    CBLINK2 = '\33[6m'
    CSELECTED = '\33[7m'

    CBLACK = '\33[30m'
    CRED = '\33[31m'
    CGREEN = '\33[32m'
    CYELLOW = '\33[33m'
    CBLUE = '\33[34m'
    CVIOLET = '\33[35m'
    CBEIGE = '\33[36m'
    CWHITE = '\33[37m'

    CBLACKBG = '\33[40m'
    CREDBG = '\33[41m'
    CGREENBG = '\33[42m'
    CYELLOWBG = '\33[43m'
    CBLUEBG = '\33[44m'
    CVIOLETBG = '\33[45m'
    CBEIGEBG = '\33[46m'
    CWHITEBG = '\33[47m'

    CGREY = '\33[90m'
    CRED2 = '\33[91m'
    CGREEN2 = '\33[92m'
    CYELLOW2 = '\33[93m'
    CBLUE2 = '\33[94m'
    CVIOLET2 = '\33[95m'
    CBEIGE2 = '\33[96m'
    CWHITE2 = '\33[97m'

    CGREYBG = '\33[100m'
    CREDBG2 = '\33[101m'
    CGREENBG2 = '\33[102m'
    CYELLOWBG2 = '\33[103m'
    CBLUEBG2 = '\33[104m'
    CVIOLETBG2 = '\33[105m'
    CBEIGEBG2 = '\33[106m'
    CWHITEBG2 = '\33[107m'


class CoordinateSystem:
    def __init__(self, X: tuple) -> None:

        self.C = []
        self.Cindices = []
        self.KroneckerDelta = []
        self.Jacobian = []
        self.metric = []
        self.inv_metric = []
        self.Christoffel = []
        self.dderivatives = []

        # What not to do
        if "x" in X or "y" in X or "z" in X:
            raise ValueError

        # Symbols setup
        self.X = tuple(sp.symbols(" ".join(X)))

        self.indices = range(len(self.X))

        self.dX = "d" + X[0]
        for i in range(1, len(self.X)):
            self.dX += " d" + X[i]
        self.dX = tuple(sp.symbols(self.dX))

        self.ddX = "dd" + X[0]
        for i in range(1, len(self.X)):
            self.ddX += " dd" + X[i]
        self.ddX = tuple(sp.symbols(self.ddX))

    def __repr__(self) -> str:
        pass

    def __str__(self) -> str:
        coorstr = []
        for i in self.Cindices:
            coorstr.append(("x", "y", "z")[i] + " = " + str(self.C[i]))
        return LogColor.CYELLOW + "\n".join(coorstr) + LogColor.CEND

    def set_coordinates(self, C: tuple) -> None:
        self.C = C
        self.Cindices = range(len(self.C))

        # Attributes of the system
        self.KroneckerDelta = np.identity(len(self.C))

        self.Jacobian = np.array([[(C[i]).diff(self.X[j]) for j in self.indices] for i in self.Cindices])

        self.metric = np.array([[sp.simplify(sum(sum(
            self.Jacobian[k][i] * self.Jacobian[l][j] * self.KroneckerDelta[k][l]
            for k in self.Cindices) for l in self.Cindices)) for j in self.indices] for i in self.indices])

        self.inv_metric = np.array(sp.Matrix(self.metric).inv())

        self.Christoffel = np.array([[[sp.simplify(sum(
            (1 / 2) * self.inv_metric[k][l] * (
                    self.metric[j][l].diff(self.X[i]) + self.metric[l][i].diff(self.X[j]) - self.metric[i][j].diff(
                self.X[l]))
            for l in self.indices)) for j in self.indices] for i in self.indices] for k in self.indices])

        self.dderivatives = np.array([self.ddX[k] + sp.simplify(sum(sum(
            self.dX[i] * self.dX[j] * self.Christoffel[k][i][j]
            for j in self.indices) for i in self.indices)) for k in self.indices])

    def print_dderiv_eqns(self) -> None:
        for i in self.indices:
            print(LogColor.CYELLOW + f"{self.X[i]}_a = {self.dderivatives[i]}" + LogColor.CEND)


def print_matrix(M: list[list[int]]) -> None:
    for line in M:
        print(" ".join(map(str, line)))


def main() -> int:
    sp.init_printing()

    system = None
    conf1 = "y"
    while conf1 == "y":

        # system pick
        while True:
            Coordinates = input(LogColor.CBLUE + "Enter coordinate system: 'polar', 'parabolic', 'bipolar', "
                                                 "'2-sphere', 'cylindrical', 'spherical'\n" + LogColor.CEND)

            match Coordinates:
                case "polar":
                    system = CoordinateSystem(("r", "t"))
                    system.set_coordinates((system.X[0] * sp.cos(system.X[1]), system.X[0] * sp.sin(system.X[1])))
                    break

                case "parabolic":
                    system = CoordinateSystem(("u", "v"))
                    system.set_coordinates((system.X[0] ** 2 - system.X[1] ** 2, 2 * system.X[0] * system.X[1]))
                    break

                case "bipolar":
                    print("This one takes time...")
                    system = CoordinateSystem(("t", "s"))
                    system.set_coordinates((sp.sin(system.X[0] + system.X[1]) / sp.sin(system.X[0] - system.X[1]),
                                            (2 * sp.sin(system.X[0]) * sp.sin(system.X[1])) / sp.sin(
                                                system.X[0] - system.X[1])))
                    break

                case "2-sphere":
                    system = CoordinateSystem(("f", "t"))
                    system.set_coordinates((sp.sin(system.X[0]) * sp.cos(system.X[1]),
                                            sp.sin(system.X[0]) * sp.sin(system.X[1]), sp.cos(system.X[0])))
                    break

                case "cylindrical":
                    system = CoordinateSystem(("r", "t", "h"))
                    system.set_coordinates(
                        (system.X[0] * sp.cos(system.X[1]), system.X[0] * sp.sin(system.X[1]), system.X[2]))
                    break

                case "spherical":
                    system = CoordinateSystem(("r", "f", "t"))
                    system.set_coordinates((system.X[0] * sp.sin(system.X[1]) * sp.cos(system.X[2]),
                                            system.X[0] * sp.sin(system.X[1]) * sp.sin(system.X[2]),
                                            system.X[0] * sp.cos(system.X[1])))
                    break

                case _:
                    print("What now nigga? Try again please.")

        print(LogColor.CGREY + "Coordinate transformation:" + LogColor.CEND)
        print(system)
        print(LogColor.CGREY + "Second derivative components:" + LogColor.CEND)
        system.print_dderiv_eqns()

        # return confirmation
        while True:
            conf1 = input(LogColor.CBLUE + "Try again? y/n\n" + LogColor.CEND)
            if conf1 == "y" or conf1 == "n":
                break
            else:
                print("Unrecognised input.")

    # end greeting
    print(LogColor.CBLUE + "Have a nice day my neighbor" + LogColor.CEND, end="")
    return 0


if __name__ == '__main__':
    main()
