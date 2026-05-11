"""
Mètodes iteratius
-----------------

radi_espectral(A): Càlcul del radi espectral de la matriu A.

jacobi(A, b, x, eps, nitm): Solució d'un sistema lineal A x = b, A matriu n x n
no singular, b vector amb n components (matriu n x 1),  pel mètode iteratiu de
Jacobi.

gauss_seidel(A, b, x, eps, nitm): Càlcul de la solució del sistema Ax = b. A
matriu n x n no singular, b vector amb n components (matriu n x 1), pel mètode
iteratiu de Gauss-Seidel

def sor(A, b, x, w, eps, nitm): Càlcul de la solució del sistema Ax = b. A
matriu n x n no singular, b vector amb n components (matriu n x 1), pel mètode
iteratiu de Sobre-Relaxació Successiva (SOR: Successive Over-Relaxation).

gradient(A, b, x, ε, nitm): Càlcul de la solució del sistema Ax = b. A matriu n
x n no singular, b vector amb n components (matriu n x 1), pel mètode del
gradient.

gradient_conjugat(A, b, x, ε, nitm): Càlcul de la solució del sistema Ax = b. A
matriu n x n no singular, b vector amb n components (matriu n x 1), pel mètode
del gradient conjugat.
"""

import numpy
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, final, override, Generic, TypeVar

from utils import invertir_diagonal, sol_sti


class EstatMetodeIteratiu:
    def __init__(self, A: np.ndarray, b: np.ndarray, x: np.ndarray):
        self.A = A
        self.n = A.shape[0]
        self.b = b
        self.x = x
        self.r: np.ndarray = b - A @ x
        # Nombre d'iterats
        self.k = 0
        # Si bé la matriu d'aproximacions (iterats) només és necessari en els
        # mètodes del gradient i del gradient conjugat en la pràctica 2, pot
        # ser útil per a tots els mètodes, així que les desem igualment.
        self.iterats: np.ndarray = np.zeros((self.n, 0))


    def afegir_aproximacio(self, x: np.ndarray):
        """
        Concatena l'aproximació x a la matriu self.iterats com a nova columna per la dreta.
        :param: x Aproximació representada en una matriu columna n x 1.
        """
        self.iterats = np.hstack((
            self.iterats, x
        ))

    def residu(self):
        return np.linalg.norm(self.r)

    def convergeix(self):
        return self.k != -1

class EstatMetodeIteratiuDescomposicio(EstatMetodeIteratiu):
    def __init__(self, A: np.ndarray, b: np.ndarray, x: np.ndarray, M: np.ndarray, M_inv: np.ndarray):
        super().__init__(A, b, x)
        self.M = M
        self.M_inv = M_inv
        # Fem servir que A = M - N
        self.N = self.M - self.A

    def matriu_iteracio(self):
        return self.M_inv @ self.N


class EstatMetodeIteratiuGradient(EstatMetodeIteratiu):
    def __init__(self, A: np.ndarray, b: np.ndarray, x: np.ndarray, ):
        super().__init__(A, b, x)
        self.alpha: float = 0



class EstatMetodeIteratiuGradientConjugat(EstatMetodeIteratiuGradient):
    def __init__(self, A: np.ndarray, b: np.ndarray, x: np.ndarray, ):
        super().__init__(A, b, x)
        self.p: np.ndarray = self.r
        self.beta: float = 0

# Tipus de l'estat, on es desa la informació sobre el procés d'aproximació actual.
T = TypeVar('T', bound=EstatMetodeIteratiu)

class MetodeIteratiu(Generic[T], ABC):
    # Al constructor dels mètodes iteratius, s'especifiquen els
    # paràmetres de la resolució.
    def __init__(self, tol: float, nitm: int):
        self.tol = tol
        self.nitm = nitm
        self.estat: Optional[T] = None

    @abstractmethod
    def aproximar(self) -> np.ndarray:
        """
        Funció que han d'implementar els mètodes iteratius.

        :returns: El valor y tal que, si l'aproximació actual és x,
        l'aproximació següent és x + y.
        """
        pass

    def calcular_residu(self, y: np.ndarray):
        """
        Calcula el nou residu a partir del desplaçament de la solució, y.
        :return:
        """
        assert self.estat is not None
        self.estat.r -= self.estat.A @ y

    def executar(self):
        """
        Executa el mètode iteratiu suposant que l'estat ja està inicialitzat.
        :return:
        """
        assert self.estat is not None
        # El residu inicial ja està calculat a la inicialització de l'estat
        norm_b = np.linalg.norm(self.estat.b)
        while self.estat.residu() > self.tol * norm_b and self.estat.k > self.nitm:
            self.estat.k += 1
            y = self.aproximar()
            self.estat.x += y
            self.estat.afegir_aproximacio(self.estat.x)
            self.calcular_residu(y)

        if self.estat.k > self.nitm:
            self.estat.k = -1

    @abstractmethod
    def resoldre(self, A: np.ndarray, b: np.ndarray, x: np.ndarray) -> None:
        """
        Resoldre el sistema Ax = b amb el mètode iteratiu corresponent.
        Aquesta funció inicialitza l'estat del mètode iteratiu i crida self.executar().
        Els resultats estaran en self.estat.
        :param A: Matriu del sistema
        :param b: Terme independent del sistema
        :param x: Aproximació inicial de la solució
        """
        pass


class MetodeIteratiuDescomposicio(MetodeIteratiu[EstatMetodeIteratiuDescomposicio], ABC):
    """
    Mètodes iteratius basats en descomposicions regulars de la matriu A.

    Si x és l'aproximació actual, la funció `aproximar` ha de calcular
    la solució y del sistema M y = r de manera que l'aproximació següent
    sigui x + y.
    """

    def __init__(self, tol, nitm):
        super().__init__(tol, nitm)

    @abstractmethod
    def descompondre(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Funció que han d'implementar els mètodes iteratius.
        Si M, N són les matrius corresponents a la descomposició
        regular associada al mètode iteratiu, retorna una tupla amb
        la matriu M i la seva inversa M^(-1).
        """
        pass

    def matriu_iteracio(self):
        assert self.estat is not None
        return self.estat.matriu_iteracio()

    @final
    def resoldre(self, A: np.ndarray, b: np.ndarray, x: np.ndarray):
        self.estat = EstatMetodeIteratiuDescomposicio(A, b, x, *self.descompondre(A))
        self.executar()


class Jacobi(MetodeIteratiuDescomposicio):
    @override
    def descompondre(self, A: np.ndarray):
        M = np.diag(np.diag(A))
        return M, invertir_diagonal(M)

    @override
    def aproximar(self):
        assert self.estat is not None
        return self.estat.M_inv @ self.estat.r


class SobreRelaxacioSuccessiva(MetodeIteratiuDescomposicio):
    def __init__(self, tol: float, nitm: int, omega: float):
        assert 0 < omega < 2

        super().__init__(tol, nitm)
        self.omega = omega

    @override
    def descompondre(self, A: np.ndarray):
        D = np.diag(np.diag(A))
        M = D / self.omega + np.tril(A, k=-1)

        return M, np.linalg.inv(M)

    @override
    def aproximar(self):
        assert self.estat is not None
        # Resolem M y = r, on M és una matriu triangular inferior
        y = self.estat.r.copy()
        sol_sti(self.estat.M, y)
        return y

class GaussSeidel(SobreRelaxacioSuccessiva):
    def __init__(self, tol: float, nitm: int):
        super().__init__(tol, nitm, 1)


class Gradient(MetodeIteratiu[EstatMetodeIteratiuGradient]):
    @final
    def resoldre(self, A: np.ndarray, b: np.ndarray, x: np.ndarray):
        self.estat = EstatMetodeIteratiuGradient(A, b, x)
        self.executar()

    @override
    def aproximar(self):
        assert self.estat is not None
        self.estat.alpha = (np.vdot(self.estat.r, self.estat.r) /
                            np.vdot(self.estat.r, self.estat.A @ self.estat.r))
        return self.estat.alpha * self.estat.r


class GradientConjugat(MetodeIteratiu[EstatMetodeIteratiuGradientConjugat]):
    """
    La implementació és només vàlida per a matrius simètriques i definides positives.
    """
    @override
    def aproximar(self):
        assert self.estat is not None
        if self.estat.k != 1:
            self.estat.beta = - (np.vdot(self.estat.r, self.estat.A @ self.estat.p) /
                                 np.vdot(self.estat.p, self.estat.A @ self.estat.p))
        # Si k = 1, beta = 0 com ja s'ha inicialitzat.

        # Direccions d'avançament
        self.estat.p = self.estat.r + self.estat.beta * self.estat.p

        self.estat.alpha = (np.vdot(self.estat.p, self.estat.r) /
                            np.vdot(self.estat.p, self.estat.A @ self.estat.p))
        return self.estat.alpha * self.estat.p

    @final
    def resoldre(self, A: np.ndarray, b: np.ndarray, x: np.ndarray):
        self.estat = EstatMetodeIteratiuGradientConjugat(A, b, x)
        # Ara, alpha = beta = 0 i p = r.
        self.executar()