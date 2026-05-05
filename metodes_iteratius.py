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
from typing import Optional, Tuple, final, override

from utils import invertir_diagonal


class MetodeIteratiu(ABC):
    # Al constructor dels mètodes iteratius, s'especifiquen els
    # paràmetres de la resolució.
    def __init__(self, tol: np.float64, nitm: int):
        self.tol = tol
        self.nitm = nitm

    @abstractmethod
    def aproximar(self, A: np.ndarray, b: np.ndarray, x: np.ndarray, r: np.ndarray) -> np.ndarray:
        """
        Funció que han d'implementar els mètodes iteratius.
        """
        pass
        
    def resoldre(self, A: np.ndarray, b: np.ndarray, x: np.ndarray):
        nit = 0
        r = b - A @ x
        while np.linalg.norm(r) > self.tol * np.linalg.norm(b):
            nit += 1
            if nit > self.nitm:
                nit = -1
                break

            x = self.aproximar(A, b, x, r)
            r = b - A @ x

        return x, np.linalg.norm(r), nit

class MetodeIteratiuDescomposicio(MetodeIteratiu, ABC):
    """
    Mètodes iteratius basats en descomposicions regulars de la matriu A.
    """

    def __init__(self, tol, nitm):
        super().__init__(tol, nitm)
        self.M_inv: Optional[np.ndarray] = None
        self.N: Optional[np.ndarray] = None
  
    @abstractmethod
    def descompondre(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Funció que han d'implementar els mètodes iteratius.
        Si M, N són les matrius corresponents a la descomposició
        regular associada al mètode iteratiu, es retorna una
        tupla amb M^(-1) i N.
        """
        pass

    @override
    def aproximar(self, A, b, x, r):
        return self.M_inv @ self.N @ x + self.M_inv @ b

    @final
    def resoldre(self, A: np.ndarray, b: np.ndarray, x: np.ndarray):
        self.M_inv, self.N = self.descompondre(A)
        return super().resoldre(A, b, x)

class Jacobi(MetodeIteratiuDescomposicio):
    @override
    def descompondre(self, A: np.ndarray):
        M = np.diag(np.diag(A))
        N = A - M
        return invertir_diagonal(M), N

class SobreRelaxacioSuccessiva(MetodeIteratiuDescomposicio):
    def __init__(self, tol: np.float64, nitm: int, omega: np.float64):
        assert 0 <= omega <= 2

        super().__init__(tol, nitm)
        self.omega = omega
    
    @override
    def descompondre(self, A: np.ndarray):
        D = np.diag(np.diag(A))
        E = - np.tril(A, k=-1)
        F = - np.triu(A, k=1)

        M = D / self.omega - E
        N = (1 - self.omega) / self.omega * D + F

        return np.linalg.inv(M), F

    @staticmethod
    def trobar_omega_optima(A: np.ndarray, b: np.ndarray, x: np.ndarray, tol: np.float64, nitm: int):

        omega = np.linspace(0.047, 1.60, 200, dtype = np.float64).reshape(-1, 1)
        nit = np.zeros(len(omega), dtype =np.int64).reshape(-1, 1)

        rho = omega.copy()

        for i in range(len(omega)):
            x = np.zeros(len(b)).reshape(-1,1)

            # Matriu d'iteració
            M = np.diag(np.diag(A)) / omega[i] + np.tril(A, k = -1)
            matriu_iteracio = np.linalg.inv(M) @ (M - A)

            # Càlcul del radi espectral de la matriu d'iteració
            rho[i] = radi_espectral(matriu_iteracio)

            metode = SobreRelaxacioSuccessiva(tol, nitm, omega[i])
            x, r, it = metode.resoldre(A, b, x)

            print(f"i: {i:3d}, " +
                  f"omega: {omega[i][0]:7.4f}, " +
                  f"rho: {rho[i][0]:7.4f}, " +
                  f"num. iterats: {it:4d}")
            if it != -1:
                nit[i] = it
            else:
                nit[i] = nitm

        nit = nit / max(nit)

class GaussSeidel(SobreRelaxacioSuccessiva):
    def __init__(self, tol: np.float64, nitm: int):
        super().__init__(tol, nitm, 1)
    

class Gradient(MetodeIteratiu):
    @override
    def aproximar(self, A, b, x, r):
        alpha_k = np.dot(r, r) / np.dor(r, A @ r)
        return x + alpha_k * r

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
    return np.max(np.fabs(eigvals))



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


