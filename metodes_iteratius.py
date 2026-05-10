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
        self.r = b - A @ x
        self.k = 0

    def residu(self):
        return np.linalg.norm(self.r)

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

        Retorna el valor y tal que, si l'aproximació actual és x,
        l'aproximació següent és x + y.
        """
        pass

    def calcular_residu(self, y: np.ndarray):
        """
        Calcula el nou residu a partir del desplaçament de la solució y.
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
        while self.estat.residu() > self.tol * np.linalg.norm(self.estat.b):
            self.estat.k += 1
            if self.estat.k > self.nitm:
                self.estat.k = -1
                break

            y = self.aproximar()
            self.estat.x += y
            self.calcular_residu(y)

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
        E = - np.tril(A, k=-1)

        M = D / self.omega - E

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


# Mètodes iteratius

def radi_espectral(A):
    """radi_espectral(A): Càlcul del radi espectral de la matriu A.

    Def. Sigui A una matriu n x n. El radi espectral de A és el mòdul del valor propi de A de mòdul màxim.

    Input
    -----
    A : Array n x n float64
        Matriu n x n de la qual es calcula el radi espectral.

    Output
    ------
    float64
        Radi espectral de la matriu A.

    Nota: podeu fer servir la funció eigvals. Recordeu que cal importar-la:
    from numpy.linalg import eigvals
    """
    eigvals = numpy.linalg.eigvals(A)
    return np.max(np.abs(eigvals))


# Mètode de Jacobi
def jacobi(A, b, x, eps, nitm):
    """
    Solució d'un sistema lineal A x = b, A matriu n x n no singular, b vector
    amb n components (matriu n x 1),  pel mètode iteratiu de Jacobi.

    Input:
        A (n x n float64): Matriu del sistema.
        b (n x 1 float64): Terme independent del sistema.
        x (n x 1 float64): Aproximació inicial de la solució.
        eps (float64): error relatiu (precisió) demanada. Les iteracions del
                      mètode s'aturen quan ||r|| = ||b - A x|| <= eps * ||b||.
        nitm (int64): màxim nombre d'iteracions permeses.

    Output:
        x (n x 1 float 64): Última aproximació trobada de la soluvió.
        r (float 64): norma del residu de l'última aproximació calulada.
        nit (int64): nombre d'iteracions del mètode que s'han dut a terme per
                     assolir la precisió demanada.
    """
    return Jacobi(eps, nitm).resoldre(A, b, x)


# Mètode de Gauss-Siedel
def gauss_seidel(A, b, x, eps, nitm):
    """
    Càlcul de la solució del sistema Ax = b. A matriu n x n no singular, b
    vector amb n components (matriu n x 1), pel mètode iteratiu de Gauss-Seidel

    Input
    A : Array n x n float64
        Matriu del sistema A x = b, A és una matriu n x n.
    b : Array n x 1 float64
        Terme independent del sistema A x = b. b és un vector amb n components (matriu n x 1)
    x : Array n x 1 float64
        Aproximació inicial de la solució. És un vector amb n components (matriu n x 1). Llevat no es conegui alguna informació sobre la solució podem agafar x = 0.
    eps : float64
        Error relatiu (precisió) demanada. Les iteracions del mètode s'aturen quan ||r|| = ||b - Ax|| <= eps * ||b||, i es torna l'ultima aproximació calculada de la solució.
    nitm : int64
        Nombre màxim d'iteracions permeses.

    Output
    x : Array n x 1 float 64
        Última aproximació trobada de la solució del sistema.
    norma_residu : int64
        Norma 2 del residu de l'última aproximació trobada de la solució del
        sistema.
    nit: int64
        Nombre d'iteracions necessàries per assolir la precisió demanada en
        l'aproximació de la solució del sistema. Si no hi ha convergència en el
        nombre màxim d'iteracions, nit = -1.

    Nota: per resoldre el sistema triangular inferior, heu de fer servir la
    funció sol_sti que ja teniu programada.
    """
    # Escriviu aquí el vostre codi
    return x, norma_residu, nit


# Mètode de Sobre-Relaxació Successiva (SOR)
def sor(A, b, x, w, eps, nitm):
    """
    Càlcul de la solució del sistema Ax = b. A matriu n x n no singular, b
    vector amb n components (matriu n x 1), pel mètode iteratiu de Sobre-Relaxació Successiva (SOR: Successive Over-Relaxation).

    Input
    A : Array n x n float64
        Matriu del sistema A x = b, A és una matriu n x n.
    b : Array n x 1 float64
        Terme independent del sistema A x = b. b és un vector amb n components (matriu n x 1)
    x : Array n x 1 float64
        Aproximació inicial de la solució. És un vector amb n components (matriu n x 1). Llevat no es conegui alguna informació sobre la solució podem agafar x = 0.
    w : float64
        Paràmetre de sobre-ralaxació.
    eps : float64
        Error relatiu (precisió) demanada. Les iteracions del mètode s'aturen quan ||r|| = ||b - Ax|| <= eps * ||b||, i es torna l'ultima aproximació calculada de la solució.
    nitm : int64
        Nombre màxim d'iteracions permeses.

    Output
    x : Array n x 1 float 64
        Última aproximació trobada de la solució del sistema.
    norma_residu : float64
        Norma del residu de l'última aproximació trobada de la solució del
        sistema.
    nit: int64
        Nombre d'iteracions necessàries per assolir la precisió demanada en
        l'aproximació de la solució del sistema. Si no hi ha convergència en el nombre màxim d'iteracions, nit = -1.

    Nota: per resoldre el sistema triangular inferior, heu de fer servir la
    funció sol_sti que ja teniu programada.
    """
    # Escriviu aquí el vostre codi
    return x, norma_residu, nit


# Mètode del gradient
def gradient(A, b, x, ε, nitm):
    """
    Càlcul de la solució del sistema Ax = b. A matriu n x n no singular, b
    vector amb n components (matriu n x 1), pel mètode del gradient.

    Input
    A : Array n x n float64
        Matriu del sistema A x = b, A és una matriu n x n.
    b : Array n x 1 float64
        Terme independent del sistema A x = b. b és un vector amb n components (matriu n x 1)
    x : Array n x 1 float64
        Aproximació inicial de la solució. És un vector amb n components (matriu n x 1). Llevat no es conegui alguna informació sobre la solució podem agafar x = 0.
    epsilon : float64
        Error relatiu (precisió) demanada. Les iteracions del mètode s'aturen
        quan ||r|| = ||b - Ax|| <= eps * ||b||, i es torna l'ultima aproximació
        calculada de la solució.
    nitm : int64
        Nombre màxim d'iteracions permeses.

    Output
    x : Array n x 1 float 64
        Última aproximació trobada de la solució del sistema.
    norma_residu : float64
        Norma del residu de l'última aproximació trobada de la solució del
        sistema.
    nit: int64
        Nombre d'iteracions necessàries per assolir la precisió demanada en
        l'aproximació de la solució del sistema. Si no hi ha convergència en el nombre màxim d'iteracions, nit = -1.
    """
    # Escriviu aquí el vostre codi
    return x, norma_residu, nit


# Mètode del gradient conjugat
def gradient_conjugat(A, b, x, ε, nitm):
    """
    Càlcul de la solució del sistema Ax = b. A matriu n x n no singular, b
    vector amb n components (matriu n x 1), pel mètode del gradient conjugat.

    Input
    A : Array n x n float64
        Matriu del sistema A x = b, A és una matriu n x n.
    b : Array n x 1 float64
        Terme independent del sistema A x = b. b és un vector amb n components (matriu n x 1)
    x : Array n x 1 float64
        Aproximació inicial de la solució. És un vector amb n components (matriu n x 1). Llevat no es conegui alguna informació sobre la solució podem agafar x = 0.
    epsilon: float64
        Error relatiu (precisió) demanada. Les iteracions del mètode s'aturen
        quan ||r|| = ||b - Ax|| <= eps * ||b||, i es torna l'ultima aproximació
        calculada de la solució.

    Output
    x : Array n x 1 float 64
        Última aproximació trobada de la solució del sistema.
    norma_residu : float64
        Norma del residu de l'última aproximació trobada de la solució del
        sistema.
    nit: int64
        Nombre d'iteracions necessàries per assolir la precisió demanada en
        l'aproximació de la solució del sistema.
    """
    # Escriviu aquí el vostre codi
    return x, norma_residu, nit
