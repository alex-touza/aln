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

class ContenidorMatriu:
    def __init__(self, A: np.ndarray):
        self.n = A.shape[1]
        self.A = A

class Descomposicio(ABC, ContenidorMatriu):
    def __init__(self, A: np.ndarray):
        super().__init__(A)
        assert A.shape[1] == self.n

    @abstractmethod
    def descompondre(self) -> None: pass

    @abstractmethod
    def residu_descomposicio(self) -> np.floating: pass


class MetodeDirecte(ABC, ContenidorMatriu):
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
        return np.linalg.norm(self.b - self.A @ self.x)

class DescomposicioLU(Descomposicio):
    def __init__(self, A: np.ndarray):
        super().__init__(A)
        self.p: None | np.ndarray = None
        self.L: None | np.ndarray = None
        self.U: None | np.ndarray = None

    def pivotar(self, p: np.ndarray, k: int):
        """
        Aplica un tipus de pivotatge al pas k-èssim.

        Per defecte, no es fa res. Classes derivades de la classe `FactoritzacioLU`
        han d'implementar aquest mètode.

        :param p: Vector de permutacions que guarda les reordenacions de files de la
                    matriu, de llargada igual al nombre de files.
        :param k: Índex de la columna actual. És la posició del pivot en aquesta iteració.
        :return: No retorna cap valor. El càlcul és aplicat in-place sobre la matriu self.A.
        """
        pass

    def residu_descomposicio(self) -> np.floating:
        assert self.L is not None
        assert self.U is not None
        A_ = self.A if self.p is None else self.A[self.p[:]]

        return np.linalg.norm(A_ - self.L @ self.U)

    def descompondre(self):
        self.p = np.ndarray(range(self.n))
        assert self.p is not None

        for k in range(self.n):
            self.pivotar(self.p, k)
            for i in range(k + 1, self.n):

                self.A[i][k] = self.A[i][k] / self.A[k][k]
                for j in range(k + 1, self.n):
                    self.A[i][j] -= self.A[i][k] * self.A[k][j]

class DescomposicioLUParcial(DescomposicioLU):
    @override
    def pivotar(self, p: np.ndarray, k: int):
        # Valor màxim dels elements a la columna k, de la diagonal cap a baix.
        # Fem això en lloc d'un bucle perquè és molt més ràpid.
        r = np.argmax(self.A[k:self.n, k])

        p[[k, r]] = p[[r, k]]
        self.A[[k, r]] = self.A[[r, k]]

class DescomposicioLUParcialEsglaonat(DescomposicioLU):
    @override
    def pivotar(self, p: np.ndarray, k: int):
        r = 0
        max_val = 0
        for i in range(k, self.n):
            s = max(abs(self.A[i][j]) for j in range(k, self.n))

            act_val = abs(self.A[i][k]) / s
            if act_val > max_val:
                r = i
                max_val = act_val

        p[[k, r]] = p[[r, k]]
        self.A[[k, r]] = self.A[[r, k]]


class DescomposicioLDLT(MetodeDirecte, Descomposicio):
    def __init__(self, A: np.ndarray):
        self.L: None | np.ndarray = None
        self.D: None | np.ndarray = None
        super().__init__(A)

    @override
    def descompondre(self):
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
    def residu_descomposicio(self) -> np.floating:
        assert self.D is not None
        assert self.L is not None
        return np.linalg.norm(self.A - self.L @ self.D @ self.L.T)

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


class DescomposicioQR(Descomposicio):
    def __init__(self, A: np.ndarray):
        super().__init__(A)
        self.Q: None | np.ndarray = None
        self.R: None | np.ndarray = None

    @override
    def residu_descomposicio(self) -> np.floating:
        assert self.Q is not None
        assert self.R is not None
        return np.linalg.norm(self.A - self.Q @ self.R)

    def residu_ortogonalitat(self) -> np.floating:
        assert self.Q is not None
        assert self.R is not None
        return np.linalg.norm(np.eye() - self.Q @ self.R)

class Householder(DescomposicioQR):
    def __init__(self, A: np.ndarray, tol: float = 1.0e-14):
        super().__init__(A)
        self.tol = tol