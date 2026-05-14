from abc import ABC, abstractmethod
from typing import override

import numpy as np

from utils import sol_sti, sol_sts, invertir_diagonal


class Factoritzacio(ABC):
    def __init__(self, A: np.ndarray):
        self.m, self.n = A.shape
        self.A = A
        # Còpia per calcular el residu després de la factorització.
        self.A_original = A.copy()

    @abstractmethod
    def factoritzar(self) -> None: pass

    def partir(self) -> None:
        """
        Funció que agafa la matriu self.A modificada per self.factoritzar() i
        desa en els camps corresponents les matrius de la factorització. Per
        defecte, no fa res, així que no fa falta implementar-la si l'algorisme
        de factorització ja desa en variables separades les matrius.
        """
        pass

    @abstractmethod
    def residu_factoritzacio(self) -> np.floating: pass


class MetodeDirecte(Factoritzacio, ABC):
    def __init__(self, A: np.ndarray):
        super().__init__(A)
        self.x: None | np.ndarray = None
        self.b: None | np.ndarray = None

    @abstractmethod
    def resoldre(self, b: np.ndarray) -> None:
        """
        Resol el sistema lineal amb el terme independent `b` i en deixa
        la solució en self.x.
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

    @override
    def resoldre(self, b: np.ndarray):
        self.b = b

        assert self.b is not None
        assert self.L is not None
        assert self.U is not None
        # Resoldre Lz = b, ignorant permutacions de moment
        z = self.b.copy()
        # No assumim 1s a la diagonal de L, perquè hi ha algorismes que no ho garanteixen
        sol_sti(self.L, z)

        # Resoldre Ux = z
        self.x = z
        assert self.x is not None
        sol_sts(self.U, self.x)

    @override
    def residu_solucio(self) -> np.floating:
        assert self.b is not None
        return np.linalg.norm(self.b - self.A_original @ self.x)
    
    def partir_l_estr(self):
        """
        Obté les matrius L i U suposant que L està continguda en
        la part estrictament inferior de A.
        """
        self.L = np.tril(self.A, k=-1) + np.eye(self.n)
        self.U = np.triu(self.A)

    def partir_u_estr(self):
        """
        Obté les matrius L i U suposant que U està continguda en
        la part estrictament superior de A.
        """
        self.L = np.tril(self.A)
        self.U = np.triu(self.A, k=1) + np.eye(self.n)

class FactoritzacioLUCompacta(FactoritzacioLU, ABC):
    def residu_factoritzacio(self) -> np.floating:
        assert self.L is not None
        assert self.U is not None

        return np.linalg.norm(self.A_original - self.L @ self.U)

class Doolittle(FactoritzacioLUCompacta):
    """
    Calcula la factorització A = L U^T amb el mètode de Doolittle.
    """
    def factoritzar(self):         
        for i in range(1, self.n):
            self.A[i][0] /= self.A[0][0]

        for k in range(1, self.n):
            for j in range(k, self.n):
                s = 0
                for p in range(k):
                    s += self.A[k][p] * self.A[p][j]

                self.A[k][j] -= s
            for i in range(k + 1, self.n):
                s = 0
                for p in range(k):
                    s += self.A[i][p] * self.A[p][k]
                self.A[i][k] = (self.A[i][k] - s) / self.A[k][k]
    
    def partir(self):
        self.partir_l_estr()

class Crout(FactoritzacioLUCompacta):
    """
    Calcula la factorització A = L U^T amb el mètode de Crout.
    """
    def factoritzar(self):
        for j in range(1, self.n):
            self.A[0][j] /= self.A[0][0]

        for k in range(1, self.n):
            for i in range(k, self.n):
                s = 0
                for p in range(k):
                    s += self.A[i][p] * self.A[p][k]

                self.A[i][k] -= s
            for j in range(k + 1, self.n):
                s = 0
                for p in range(k):
                    s += self.A[k][p] * self.A[p][j]
                self.A[k][j] = (self.A[k][j] - s) / self.A[k][k]
    
    def partir(self):
        self.partir_u_estr()

class FactoritzacioLUGaussiana(FactoritzacioLU):
    """
    Calcula la factorització A = L U, per defecte sense pivotatge.
    """
    def __init__(self, A: np.ndarray, pivotatge: bool = False):
        """
        :param: A La matriu del sistema.
        :param: pivotatge Si el mètode fa servir pivotatge.
        """
        super().__init__(A)
        self.pivotatge = pivotatge
        self.p: None | np.ndarray = None
        self.b_original: None | np.ndarray = None

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
        if self.pivotatge:
            self.p = np.array(list(range(self.n)))
            assert self.p is not None

        for k in range(self.n):
            if self.pivotatge:
                self.pivotar(k)
            for i in range(k + 1, self.n):

                self.A[i][k] = self.A[i][k] / self.A[k][k]
                for j in range(k + 1, self.n):
                    self.A[i][j] -= self.A[i][k] * self.A[k][j]

    @override
    def partir(self):
        self.partir_l_estr()

    def resoldre(self, b: np.ndarray):
        if self.pivotatge:
            assert self.p is not None
            # Desem el terme independent sense permutar per calcular
            # el residu després
            self.b_original = b.copy()
            
            super().resoldre(b[self.p[:]])
        else:
            super().resoldre(b)

    @override
    def residu_solucio(self) -> np.floating:
        b = self.b_original if self.pivotatge else self.b
        assert b is not None
        assert self.x is not None
        return np.linalg.norm(b - self.A_original @ self.x)
    
    @override
    def residu_factoritzacio(self) -> np.floating:
        assert self.L is not None
        assert self.U is not None
        A_ = self.A_original if self.p is None else self.A_original[self.p[:]]

        return np.linalg.norm(A_ - self.L @ self.U)


class FactoritzacioLUParcial(FactoritzacioLUGaussiana):
    """
    Calcula la factorització A = L U amb pivotatge parcial.
    """
    def __init__(self, A: np.ndarray):
        super().__init__(A, pivotatge=True)
    
    @override
    def pivotar(self, k: int):
        assert self.p is not None
        # Valor màxim dels elements a la columna k, de la diagonal cap a baix.
        # Fem això en lloc d'un bucle perquè és molt més ràpid.
        r = np.argmax(np.abs(self.A[k:self.n, k])) + k

        self.p[[k, r]] = self.p[[r, k]]
        self.A[[k, r]] = self.A[[r, k]]

class FactoritzacioLUParcialEsglaonat(FactoritzacioLUGaussiana):
    """
    Calcula la factorització A = L U amb pivotatge parcial esglaonat.
    """
    def __init__(self, A: np.ndarray):
        super().__init__(A, pivotatge=True)
    
    @override
    def pivotar(self, k: int):
        assert self.p is not None
        r = 0
        max_val = 0
        for i in range(k, self.n):
            s = max(abs(self.A[i][j]) for j in range(k, self.n))
            if s == 0:
                continue
            act_val = abs(self.A[i][k]) / s
            if act_val > max_val:
                r = i
                max_val = act_val

        self.p[[k, r]] = self.p[[r, k]]
        self.A[[k, r]] = self.A[[r, k]]


class FactoritzacioLDLT(MetodeDirecte):
    """
    Calcula la factorització A = L D^(-1) L^T, on L és una matriu triangular inferior i
    D és una matriu diagonal.
    A ha de ser simètrica.
    """
    def __init__(self, A: np.ndarray):
        self.L: None | np.ndarray = None
        self.D: None | np.ndarray = None
        super().__init__(A)

    @override
    def factoritzar(self):
        n = self.A.shape[0]
        assert self.A.shape[1] == n

        # A[1:][0] /= A[0][0]
        for i in range(1, n):
            self.A[i][0] /= self.A[0][0]
        for k in range(1, n):
            self.A[k][k] -= sum([
                self.A[k][r] ** 2 * self.A[r][r]
                for r in range(k)
            ])

            for i in range(k + 1, n):
                self.A[i][k] -= sum([
                    self.A[i][r] * self.A[r][r] * self.A[k][r]
                    for r in range(k)
                ])
                self.A[i][k] /= self.A[k][k]
    @override
    def partir(self):
        self.D = np.diag(np.diag(self.A))
        self.L = np.tril(self.A, k=-1) + np.eye(self.n)

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
        # Hem de resoldre LDL^T x = b

        # Resolem Ly = b
        y = self.b.copy()
        sol_sti(self.L, y)

        # Resolem L^T x = D^(-1) y
        self.x = invertir_diagonal(self.D) @ y
        assert self.x is not None
        sol_sts(self.L.T, self.x)

class FactoritzacioCholesky(MetodeDirecte):
    """
    Calcula la factorització A = L L^T, on L és una matriu triangular inferior.
    A ha de ser simètrica i definida positiva.
    """
    def __init__(self, A: np.ndarray):
        self.L: None | np.ndarray = None
        super().__init__(A)

    @override
    def factoritzar(self) -> None:
        self.A[0][0] = np.sqrt(self.A[0][0])
        for i in range(1, self.n):
            self.A[i][0] /= self.A[0][0]
        for j in range(1, self.n):
            self.A[j][j] -= sum([self.A[j][k] ** 2 for k in range(j)])
            self.A[j][j] = np.sqrt(self.A[j][j])
            for i in range(j + 1, self.n):
                self.A[i][j] -= sum([self.A[i][k] * self.A[j][k] for k in range(j)])
                self.A[i][j] /= self.A[j][j]
    @override
    def partir(self):
        self.L = np.tril(self.A)

    @override
    def resoldre(self, b):
        assert self.L is not None
        self.b = b
        # Resol L y = b
        assert self.b is not None
        y = self.b.copy()
        sol_sti(self.L, y)

        # Resolem L^T x = y
        self.x = y
        assert self.x is not None
        sol_sts(self.L.T, self.x)

    @override
    def residu_factoritzacio(self):
        assert self.L is not None
        return np.linalg.norm(self.A_original - self.L @ self.L.T)   
    


class FactoritzacioQR(Factoritzacio, ABC):
    """
    Classe abstracta per a les factoritzacions QR.
    """
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
    """
    Calcula la factorització A = Q R amb el mètode de Householder.
    """
    def __init__(self, A: np.ndarray, tol: float = 1.0e-14):
        super().__init__(A)
        self.tol = tol


    def vector(self, x: np.ndarray):
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
        # R = Q^(-1) A = Q^T A
        assert self.Q is not None # De nou, per fer feliç el type-checker
        self.R = self.Q.T @ self.A



class GramSchmidtClassic(FactoritzacioQR):
    """
    Calcula la factorització A = Q R amb el mètode de Gram-Schmidt clàssic.
    """
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
    # No cal implementar partir, ja que no ja tenim les matrius Q i R

class GramSchmidtModificat(FactoritzacioQR):
    """
    Calcula la factorització A = Q R amb el mètode de Gram-Schmidt modificat.
    """
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
    # No cal implementar partir, ja que ja tenim les matrius Q i R