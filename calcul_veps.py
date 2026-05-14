from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Generic, TypeVar, override

import numpy as np

from metodes_directes import FactoritzacioLUParcialEsglaonat


@dataclass
class ResultatMetodePotencia:
    vep_associat: np.ndarray
    """
    Vector propi associat al valor propi de mòdul més gran.
    """
    vap_dominant: np.floating
    """
    Valor propi de mòdul màxim.
    """


@dataclass
class ResultatMetodePotenciaInversa:
    vep_associat: np.ndarray
    """
    Vector propi associat al valor propi de mòdul mínim.
    """
    vap_minim: np.floating
    """
    Valor propi de mòdul mínim.
    """


@dataclass
class ResultatMetodePotenciaInversaDesplacada:
    vep_associat: np.ndarray
    """
    Vector propi associat al valor propi més proper al paràmetre q.
    """
    vap_mes_proper: np.floating
    """
    Valor propi més proper al paràmetre q.
    """




class EstatMetodePotencia:
    """
    Estat per als mètodes de la potència i la potència inversa.

    Nota: Si bé en els apunts de l'assignatura el mètode de la potència inversa fa servir x_k
    en lloc de z_k, per simplificar el codi mantenim el nom de la variable com z_k.
    """

    def __init__(self, A: np.ndarray, z0: np.ndarray):
        self.A = A
        self.n = A.shape[0]
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


class EstatMetodePotenciaInversa(EstatMetodePotencia):
    def __init__(self, A: np.ndarray, z0: np.ndarray):
        super().__init__(A, z0)
        self.factoritzacio = FactoritzacioLUParcialEsglaonat(A)
        self.factoritzacio.factoritzar()
        self.factoritzacio.partir()

    def resoldre(self) -> np.ndarray:
        """
        Resol el sistema lineal Ay = z amb factorització LU amb pivotatge parcial esglaonat i en
        retorna la solució y.
        """
        assert self.factoritzacio is not None
        self.factoritzacio.resoldre(self.z)
        assert self.factoritzacio.x is not None
        return self.factoritzacio.x


TResultat = TypeVar("TResultat")
TEstat = TypeVar("TEstat", bound=EstatMetodePotencia)


class MetodePotenciaBase(Generic[TEstat, TResultat], ABC):
    """
    Classe base abstracta per als mètodes de la potència i les seves variants.
    """
    def __init__(self, epsilon: float):
        self.epsilon = epsilon
        self.estat: Optional[TEstat] = None

    @abstractmethod
    def inicialitzar_estat(self, A: np.ndarray, z0: np.ndarray) -> TEstat:
        """
        Retorna un objecte de tipus TEstat, que és derivat de EstatMetodePotenciaBase,
        que representa l'estat inicial per a l'execució del mètode la potència o alguna
        de les seves variants.

        :param A: Matriu
        :param z0: Aproximació inicial
        :return: Retorna un objecte que representa l'estat inicial.
        """
        pass

    @abstractmethod
    def aproximar(self) -> np.ndarray:
        """
        A partir de l'estat actual del mètode self.estat, retorna una tupla
        amb la nova aproximació de vector propi y.
        """
        pass

    @abstractmethod
    def resultat(self) -> TResultat:
        """
        Mètode abstracte que cal implementar a les subclasses.

        :return: Retorna el resultat calculat en una instància de tipus `TResultat`.
        """
        pass

    def calcular(self, A: np.ndarray, z0: np.ndarray) -> None:
        """
        Executa el mètode de la potència o alguna de les seves variants. No retorna res,
        però actualitza el valor de self.estat i, per tant, el resultat es pot obtenir a través
        de self.resultat() després d'executar aquest mètode.

        La matriu A i el vector z0 es poden modificar dins del mètode, així que es recomana passar
        còpies d'aquests si es volen mantenir els originals sense canvis.
        """
        assert np.linalg.norm(z0, ord=np.inf) == 1
        self.estat = self.inicialitzar_estat(A, z0)

        assert self.estat is not None
        # Python no té un do-while...
        seguir = True
        while seguir:
            self.estat.y = self.aproximar()
            self.estat.z = self.estat.y / np.linalg.norm(self.estat.y)

            seguir = self.estat.residu() > self.epsilon

        return self.resultat()


class MetodePotencia(MetodePotenciaBase[EstatMetodePotencia, ResultatMetodePotencia]):
    @override
    def inicialitzar_estat(self, A: np.ndarray, z0: np.ndarray) -> EstatMetodePotencia:
        return EstatMetodePotencia(A, z0)

    @override
    def resultat(self) -> ResultatMetodePotencia:
        assert self.estat is not None
        return ResultatMetodePotencia(self.estat.y, np.linalg.norm(self.estat.y))

    @override
    def aproximar(self) -> np.ndarray:
        assert self.estat is not None
        return self.estat.A @ self.estat.z


class MetodePotenciaInversa(MetodePotenciaBase[EstatMetodePotenciaInversa, ResultatMetodePotenciaInversa]):
    @override
    def inicialitzar_estat(self, A: np.ndarray, z0: np.ndarray) -> EstatMetodePotenciaInversa:
        return EstatMetodePotenciaInversa(A, z0)

    @override
    def resultat(self) -> ResultatMetodePotenciaInversa:
        assert self.estat is not None
        return ResultatMetodePotenciaInversa(self.estat.z, 1 / np.linalg.norm(self.estat.y))

    @override
    def aproximar(self) -> np.ndarray:
        assert self.estat is not None

        # Resoldre Ay = z amb LU i retornar y
        self.estat.factoritzacio.resoldre(self.estat.z)

        assert self.estat.factoritzacio.x is not None
        return self.estat.factoritzacio.x  # y


class MetodePotenciaDesplacada(MetodePotencia):
    def __init__(self, epsilon: float, q: float):
        super().__init__(epsilon)
        self.q = q

    @override
    def inicialitzar_estat(self, A: np.ndarray, z0: np.ndarray) -> EstatMetodePotencia:
        return EstatMetodePotencia(A - self.q * np.eye(A.shape[0]), z0)

    @override
    def resultat(self) -> ResultatMetodePotencia:
        resultat_ = super().resultat()
        return ResultatMetodePotencia(resultat_.vep_associat, resultat_.vap_dominant + self.q)


class MetodePotenciaInversaDesplacada(MetodePotenciaInversa):
    def __init__(self, epsilon: float, q: float):
        super().__init__(epsilon)
        self.q = q

    @override
    def inicialitzar_estat(self, A: np.ndarray, z0: np.ndarray) -> EstatMetodePotenciaInversa:
        return EstatMetodePotenciaInversa(A - self.q * np.eye(A.shape[0]), z0)

    @override
    def resultat(self) -> ResultatMetodePotenciaInversaDesplacada: # pyright: ignore[reportIncompatibleMethodOverride]
        """
        Aquí el tipus retornat és diferent que el de la funció original en MetodePotenciaInversa.
        Això trenca el principi de substitució de Liskov, però com a la pràctica només farem servir
        aquestes classes directament i no a través de classes parents, ho ignorarem.

        :return: Un objecte de tipus ResultatMetodePotenciaInversaDesplacada que indica el valor propi
        més proper al paràmetre q i el vector propi associat a aquest valor propi.
        """
        resultat_ = super().resultat()
        return ResultatMetodePotenciaInversaDesplacada(resultat_.vep_associat, 1 / (resultat_.vap_minim + self.q))




metode = MetodePotencia(1e-8)
resultat = metode.calcular(np.array([
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
print(resultat.vap, resultat.vep_associat)

assert metode.estat is not None

print("VAP de mòdul més gran: ", np.linalg.norm(metode.estat.y))
print("VEP associat: ", metode.estat.y)
