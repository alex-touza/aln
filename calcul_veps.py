from dataclasses import dataclass
from typing import Optional, Generic, TypeVar, override

import numpy as np


@dataclass
class ResultatMetodePotencia:
    vep_associat: np.ndarray
    vap_dominant: np.floating


class EstatMetodePotencia:
    def __init__(self, A: np.ndarray, z0: np.ndarray):
        self.A = A
        self.z_: np.ndarray = z0
        self.z_prev: Optional[np.ndarray] = None
        self.y: np.ndarray = np.array([])

    def residu(self):
        assert self.z_prev is not None
        return np.linalg.norm(self.z - self.z_prev)

    @property
    def z(self):
        return self.z_

    @z.setter
    def z(self, z: np.ndarray):
        self.z_prev = self.z_.copy()
        self.z_ = z

    def resultat(self):
        return ResultatMetodePotencia(self.y, np.linalg.norm(self.y))


TEstat = TypeVar('TEstat', bound=EstatMetodePotencia)


class MetodePotencia:
    def __init__(self, epsilon: float):
        self.epsilon = epsilon
        self.estat: Optional[EstatMetodePotencia] = None

    def inicialitzar_estat(self, A: np.ndarray, z0: np.ndarray) -> EstatMetodePotencia:
        return EstatMetodePotencia(A, z0)
    
    def aproximar(self) -> np.ndarray:
        """
        A partir de l'estat actual del mètode self.estat, retorna una tupla
        amb la nova aproximació de vector propi y.
        """
        assert self.estat is not None
        return self.estat.A @ self.estat.z

    def calcular(self, A: np.ndarray, z0: np.ndarray):
        assert np.linalg.norm(z0, ord=np.inf) == 1
        self.estat = self.inicialitzar_estat(A, z0)

        assert self.estat is not None
        # Python no té un do-while...
        seguir = True
        while seguir:
            self.estat.y = self.aproximar()
            self.estat.z = self.estat.y / np.linalg.norm(self.estat.y)

            seguir = self.estat.residu() > self.epsilon

        return self.estat.resultat()
    

class MetodePotenciaInversa(MetodePotencia):
    @override
    def inicialitzar_estat(self, A: np.ndarray, z0: np.ndarray) -> np.ndarray:
        # Resoldre Ay = z amb LU i retornar y 

metode = MetodePotencia(1e-8)
metode.calcular(np.array([
    [1, 1, 1, 1],
    [1, 2, 1, 1],
    [1, 1, 3, 1],
    [1, 1, 1, 4],
]), np.array(np.array([
    [1],
    [1],
    [1],
    [1]
]
)))



assert metode.estat is not None

print("VAP de mòdul més gran: ", np.linalg.norm(metode.estat.y))
print("VEP associat: ", metode.estat.y)
