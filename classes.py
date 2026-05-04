from abc import ABC, abstractmethod, abstractstaticmethod
from typing import Generic, override

import numpy as np

from sistemes_triangulars import sol_sti, sol_sts


def invertir_diagonal(D: np.ndarray):
    elements = np.diag(D)

    for x in elements:
        if x != 0:
            x = 1 / x

    return np.diag(elements)

class Factoritzacio(ABC):
    def __init__(self, A: np.ndarray):
        self.m, self.n = A.shape
        self.A = A
        # Còpia per calcular el residu després de la factorització.
        self.A_original = A.copy()

    @abstractmethod
    def factoritzar(self) -> None: pass

    @abstractmethod
    def residu_factoritzacio(self) -> np.floating: pass


class MetodeDirecte(Factoritzacio, ABC):
    def __init__(self, A: np.ndarray):
        super().__init__(A)
        self.x: None | np.ndarray = None
        self.b: None | np.ndarray = None

    @abstractmethod
    def resoldre(self, b: np.ndarray):
        """
        Resol el sistema lineal amb el terme independent `b`.
        Aquesta funció s'ha d'executar després de `descompondre`.
        :param b: El terme independent del sistema.
        """
        pass

    def residu_solucio(self) -> np.floating:
        assert self.b is not None
        assert self.x is not None
        return np.linalg.norm(self.b - self.A_original @ self.x)

class FactoritzacioLU(MetodeDirecte, ABC):
    def __init__(self, A: np.ndarray):
        super().__init__(A)
        self.L: None | np.ndarray = None
        self.U: None | np.ndarray = None

class FactoritzacioLUCompacta(FactoritzacioLU, ABC):
    def residu_factoritzacio(self) -> np.floating:
        assert self.L is not None
        assert self.U is not None

        return np.linalg.norm(self.A_original - self.L @ self.U)


class FactoritzacioLUGaussiana(FactoritzacioLU):
    def __init__(self, A: np.ndarray):
        super().__init__(A)
        self.p: None | np.ndarray = None

    def pivotar(self, k: int):
        """
        Aplica un tipus de pivotatge al pas k-èssim, fent les permutacions corresponents
        a self.p.

        Per defecte, no es fa res. Classes derivades de la classe `FactoritzacioLU`
        han d'implementar aquest mètode.

        :param k: Índex de la columna actual. És la posició del pivot en aquesta iteració.
        :return: No retorna cap valor. El càlcul és aplicat in-place sobre la matriu self.A.
        """
        pass

    @override
    def factoritzar(self):
        print("n", self.n)
        self.p = np.array(list(range(self.n)))
        assert self.p is not None

        for k in range(self.n):
            self.pivotar(k)
            for i in range(k + 1, self.n):

                self.A[i][k] = self.A[i][k] / self.A[k][k]
                for j in range(k + 1, self.n):
                    self.A[i][j] -= self.A[i][k] * self.A[k][j]

        self.L = np.tril(self.A, k=-1) + np.eye(self.n)
        self.U = np.triu(self.A)


    def invertir_permutacio(self) -> np.ndarray:
        """
        Obté el vector permutació que desfa les permutacions fetes per self.p.
        """
        pp = np.array(list(range(self.n)))
        for i in range(self.n):
            pp[i] = np.where(self.p == pp[i])[0][0]

        return pp

    def resoldre(self, b: np.ndarray):
        self.b = b

        assert self.b is not None
        assert self.L is not None
        assert self.U is not None
        # Resoldre Lz = b, ignorant permutacions de moment
        z = self.b.copy()
        sol_sti(self.L, z)

        # Resoldre Ux = z
        self.x = z
        assert self.x is not None
        sol_sts(self.U, self.x)

        # Ara, tenim en compte les permutacions si cal
        return self.x if self.p is None else self.x[self.invertir_permutacio()[:]]

    def residu_solucio(self) -> np.floating:
        return np.linalg.norm(self.b - self.A_original @ self.x)

    @override
    def residu_factoritzacio(self) -> np.floating:
        assert self.L is not None
        assert self.U is not None
        A_ = self.A_original if self.p is None else self.A_original[self.p[:]]

        return np.linalg.norm(A_ - self.L @ self.U)


class FactoritzacioLUParcial(FactoritzacioLUGaussiana):
    @override
    def pivotar(self, k: int):
        assert self.p is not None
        # Valor màxim dels elements a la columna k, de la diagonal cap a baix.
        # Fem això en lloc d'un bucle perquè és molt més ràpid.
        r = np.argmax(np.abs(self.A[k:self.n, k])) + k

        self.p[[k, r]] = self.p[[r, k]]
        self.A[[k, r]] = self.A[[r, k]]

class FactoritzacioLUParcialEsglaonat(FactoritzacioLUGaussiana):
    @override
    def pivotar(self, k: int):
        assert self.p is not None
        r = 0
        max_val = 0
        for i in range(k, self.n):
            s = max(abs(self.A[i][j]) for j in range(k, self.n))

            act_val = abs(self.A[i][k]) / s
            if act_val > max_val:
                r = i
                max_val = act_val

        self.p[[k, r]] = self.p[[r, k]]
        self.A[[k, r]] = self.A[[r, k]]


class FactoritzacioLDLT(MetodeDirecte):
    def __init__(self, A: np.ndarray):
        self.L: None | np.ndarray = None
        self.D: None | np.ndarray = None
        super().__init__(A)

    @override
    def factoritzar(self):
        n = self.A.shape[0]
        assert self.A.shape[1] == n

        # A[1:][0] /= A[0][0]
        for i in range(2, n + 1):
            self.A[i - 1][0] = self.A[i - 1][0] / self.A[0][0]
        for k in range(2, n + 1):
            self.A[k - 1][k - 1] -= np.sum([
                self.A[k - 1][r - 1] * self.A[k - 1][r - 1] * self.A[r - 1][r - 1]
                for r in range(1, k)
            ])

            for i in range(k + 1, n + 1):
                self.A[i - 1][k - 1] -= np.sum([
                    self.A[i - 1][r - 1] * self.A[r - 1][r - 1] * self.A[k - 1][r - 1]
                    for r in range(1, k)
                ])
                self.A[i - 1][k - 1] /= self.A[k - 1][k - 1]

        self.D = np.diag(np.diag(self.A))
        self.L = np.tril(self.A, k=-1) + np.eye(n)

    @override
    def residu_factoritzacio(self) -> np.floating:
        assert self.D is not None
        assert self.L is not None
        return np.linalg.norm(self.A_original - self.L @ self.D @ self.L.T)

    @override
    def resoldre(self, b: np.ndarray):
        self.b = b
        assert self.D is not None
        assert self.L is not None
        assert self.b is not None

        # Resolem Ly = b
        y = self.b.copy()
        sol_sti(self.L, y)

        # Resolem Lt x = D^(-1) y
        self.x = invertir_diagonal(self.D) @ y
        assert self.x is not None
        sol_sts(self.L.T, self.x)

        return self.x

class FactoritzacioCholesky(MetodeDirecte):
    def __init__(self, A: np.ndarray):
        super().__init__(A)

    def factoritzar(self) -> None:
        pass


class FactoritzacioQR(Factoritzacio, ABC):
    def __init__(self, A: np.ndarray):
        super().__init__(A)
        self.Q: None | np.ndarray = None
        self.R: None | np.ndarray = None

    @override
    def residu_factoritzacio(self) -> np.floating:
        assert self.Q is not None
        assert self.R is not None
        return np.linalg.norm(self.A_original - self.Q @ self.R)

    def residu_ortogonalitat(self) -> np.floating:
        assert self.Q is not None
        assert self.R is not None
        return np.linalg.norm(np.eye(self.m) - self.Q.T @ self.Q)

class Householder(FactoritzacioQR):
    def __init__(self, A: np.ndarray, tol: float = 1.0e-14):
        super().__init__(A)
        self.tol = tol


    def vector(self, x: np.ndarray):
        x = np.asanyarray(x, dtype=np.float64)
        norm_x = np.linalg.norm(x, ord=2)

        v = - norm_x if x[0] > 0 else norm_x
        u = x.copy()
        u[0] -= v

        if np.abs(v) < self.tol: beta = 0.0
        else: beta = - 1.0 / (v * u[0])

        return beta, u

    @override
    def factoritzar(self) -> None:
        n_steps = self.n if self.m > self.n else self.n - 1

        self.Q = np.eye(self.m)

        for k in range(n_steps):
            target = self.A[k:, k]
            if np.max(np.abs(target[1:])) < self.tol: continue

            beta, u = self.vector(target)
            H = np.eye(self.m - k) - beta * np.outer(u, u)
            P_k = np.block([
                [np.eye(k),         np.zeros((k, self.m - k))  ],
                [np.zeros((self.m - k, k)),  H                   ]
            ])

            assert self.Q is not None # Per fer feliç el type-checker
            self.Q = self.Q @ P_k
            self.A = P_k @ self.A

        self.R = self.A



class GramSchmidtClassic(FactoritzacioQR):
    def factoritzar(self) -> None:
        self.R = np.zeros((self.m, self.n))
        self.Q = np.zeros((self.m, self.m))
        assert self.R is not None
        assert self.Q is not None

        self.R[0][0] = np.linalg.norm(self.A[:, 0])
        self.Q[:, 0] = self.A[:, 0] / self.R[0][0]

        for k in range(1, self.n):
            for j in range(k):
                self.R[j][k] = np.dot(self.Q[:, j], self.A[:, k])

            # axis=0 suma respecte els vectors fila de cada summand
            self.A[:, k] -= np.sum([self.R[v][k] * self.Q[:, v] for v in range(k)], axis=0)
            self.R[k][k] = np.linalg.norm(self.A[:, k])
            self.Q[:, k] = self.A[:, k] / self.R[k][k]


class GramSchmidtModificat(FactoritzacioQR):
    def factoritzar(self) -> None:
        self.R = np.zeros((self.m, self.n))
        self.Q = np.zeros((self.m, self.m))
        assert self.R is not None
        assert self.Q is not None

        self.R[0][0] = np.linalg.norm(self.A[:, 0])
        self.Q[:, 0] = self.A[:, 0] / self.R[0][0]

        for k in range(1, self.n):
            for s in range(k, self.n):
                self.R[k - 1, s] = np.dot(self.Q[:, k - 1], self.A[:, s])
                self.A[:, s] -= self.R[k - 1, s] * self.Q[:, k - 1]

            self.R[k][k] = np.linalg.norm(self.A[:, k])
            self.Q[:, k] = self.A[:, k] / self.R[k][k]

