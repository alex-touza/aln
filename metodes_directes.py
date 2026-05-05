"""
metodes_directes.py

Mètodes directes
----------------

sol_sti(A, b): Resolució d'un sistema triangular inferior

sol_sts(A, b): Resolució d'un sistema triangular superior

lu(A): Factorització LU d'una matriu quadrada A

lu_pp(A): Factorització LU d'una matriu quadrada A amb pivot parcia

lu_ppe(A): Factorització LU d'una matriu quadrada A amb pivot parcial esglaonat

doolittle(A): Factorització LU de Doolittle d'una matriu quadrada A

crout(A): Factorització LU de Crout d'una matriu quadrada A

ldlt(A): Factorització LDLT d'una matriu quadrada A

choleski(A): Factorització de Choleski d'una matriu quadrada A

sol_lu (A, b): Resolució del sistema a partir de la factorització LU de la matriu A

gs_classic(a): Factorització QR d'una matriu a mitjançant el mètode de Gram-Schmidt clàssic

gs_modificat(a): Factorització QR d'una matriu a mitjançant el mètode de Gram-Schmidt modificat

house(x, tol): Calcula el vector de Householder u i el coeficient beta per a un vector x
qr_house(a): Factorització QR d'una matriu a mitjançant el mètode de Householder
"""
from typing import override, Tuple

import numpy as np


# Solució de sistemes triangulars


# Factorització LU: LU sense pivotatge, amb pivotatge parcial i pivotatge
# parcial esglaonat

class FactoritzacioLU:
    def __init__(self, A: np.ndarray):
        self.n = A.shape[0]
        assert A.shape[1] == self.n

        self.A = A

    def pivotar(self, p: np.ndarray, k: int):
        """
        Aplica un tipus de pivotatge en un conjunt de dades matriu i vector.

        Per defecte, no es fa res. Classes derivades de la classe `FactoritzacioLU`
        han d'implementar aquest mètode.

        :param p: Vector de permutacions que guarda les reordenacions de files de la
                    matriu, de llargada igual al nombre de files.
        :param k: Índex de la columna actual. És la posició del pivot en aquesta iteració.
        :return: No retorna cap valor. El càlcul és aplicat in-place sobre la matriu self.A.
        """
        pass

    def calcular(self):
        p = np.ndarray(range(self.n))

        for k in range(self.n):
            self.pivotar(p, k)
            for i in range(k + 1, self.n):

                self.A[i][k] = self.A[i][k] / self.A[k][k]
                for j in range(k + 1, self.n):
                   self.A[i][j] -= self.A[i][k] * self.A[k][j]

        return self.A, p


class FactoritzacioLUParcial(FactoritzacioLU):
    @override
    def pivotar(self, p: np.ndarray, k: int):
        # Valor màxim dels elements a la columna k, de la diagonal cap a baix.
        # Fem això en lloc d'un bucle perquè és molt més ràpid.
        r = np.argmax(self.A[k:self.n, k])

        p[[k, r]] = p[[r, k]]
        self.A[[k, r]] = self.A[[r, k]]


class FactoritzacioLUParcialEsglaonat(FactoritzacioLU):
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


def lu(A):
    """
    lu(A): Factorització LU d'una matriu quadrada A.

    Input:
       A: Matriu quadrada d'ordre n x n, que es modificarà in-place per
       contenir les components de L i U.

    Output:
       A: modificada, de manera que:

          La part triangular estrictament inferior de A (a(i,j), 1 <=j < i <= n)
          conté els coeficients m(i,j) (1 <= j < i <= n) de la matriu L.

          La part triangular superior de A (a(i,j) (a(i,j), 1 <= i <= j <= n)
          conté els coeficients u(i,j) (1 <= i <= j <= n) de la matriu U.
    """
    factoritzacio = FactoritzacioLU(A)
    factoritzacio.calcular() # retorna p, però no cal
    return A


def lu_pp(A):
    """
    lu_pp(A): Realitza la factorització LU de la matriu A amb pivot parcial.

    Input:
       A: Matriu quadrada d'ordre n x n, que es modificarà in-place per contenir
         les components de L i U.

    Output:
       A: modificada de manera que la part triangular superior de A conté
         els elements u(i,j) (1 <=i <= j < n) de la matriu U i la part
         triangular estrictament inferior (i.e., per sota de la diagonal) de
         A conté els elements m(i,j) (1 <=j < i <= n) de la matriu L.

       p: vector de permutació que indica l'ordre de les files després del
         pivot parcial.
    """

    factoritzacio = FactoritzacioLUParcial(A)
    return factoritzacio.calcular()


def lu_ppe(A):
    """
    lu_pp(A): Realitza la factorització LU de la matriu A amb pivot parcial
              esglaonat.

    Input:
      A: Matriu quadrada d'ordre n x n, que es modificarà in-place per contenir
        les components de L i U.

    Output:
      A: modificada de manera que la part triangular superior de A conté
        els elements u(i,j) (1 <=i <= j < n) de la matriu U i la part
        triangular estrictament inferior (i.e., per sota de la diagonal) de
        A conté els elements m(i,j) (1 <=j < i <= n) de la matriu L.

      p: vector de permutació que indica l'ordre de les files després del
        pivot parcial esglaonat.
    """
    factoritzacio = FactoritzacioLUParcialEsglaonat(A)
    return factoritzacio.calcular()


# Esquemes compactes: Doolittle, Crout, LDLT i Choleski

def doolittle(A):
    """
    doolittle(A): Factorització LU de Doolittle d'una matriu quadrada A.

    Input:
       A: Matriu quadrada d'ordre n x n, que es modificarà in-place per
         contenir les components de L i U.

    Output:
       A: modificada de manera que la part triangular superior de A
         (a(i,j), 1 <=j < i <= n) conté els elements u(i,j), 1 <= i <= j <=
         n, de la matriu U i la part triangular estrictament inferior (i.e.,
         per sota de la diagonal) de A (a(i,j), 1 <=j < i <= n) conté els
         elements m(i,j), 1 <=j < i <= n, de la matriu L.
    """
    n = A.shape[0]

    for i in range(1, n):
        A[i][0] /= A[0][0]

    for k in range(1, n):
        for j in range(k, n):
            s = 0
            for p in range(k):
                s += A[k][p] * A[p][j]

            A[k][j] -= s
        for i in range(k + 1, n):
            s = 0
            for p in range(k):
                s += A[i][p] * A[p][k]
            A[i][k] = (A[i][k] - s) / A[k][k]

    return A


def crout(A):
    """
    crout(A): Factorització LU de Crout d'una matriu quadrada A.

    Input:
       A: Matriu quadrada d'ordre n x n, que es modificarà in-place per
         contenir les components de L i U.

    Output:
       A: modificada de manera que la part triangular estrictament superior de A
         (a(i,j), 1 <=j < i <= n) conté els elements u(i,j), 1 <= i < j <= n, de
         la matriu U i la part triangular inferior (i.e., per sota de la diagonal)
         de A (a(i,j), 1 <=j <= i <= n) conté els elements m(i,j), 1 <=j < i <= n,
         de la matriu L.
    """
    n = A.shape[0]

    for j in range(1, n):
        A[0][j] /= A[0][0]

    for k in range(1, n):
        for i in range(k, n):
            s = 0
            for p in range(k):
                s += A[i][p] * A[p][k]

            A[i][k] -= s
        for j in range(k + 1, n):
            s = 0
            for p in range(k):
                s += A[k][p] * A[p][j]
            A[k][j] = (A[k][j] - s) / A[k][k]

    return A


def eldlt(A):
    """
    ldlt(A): Factorització LDLT d'una matriu quadrada A.

    Input:
        A: Matriu quadrada d'ordre n x n, que es modificarà in-place per
          contenir les components de L i D.

    Output:
        A: modificada de manera que la part triangular estrictament inferior de
          A (a(i,j), 1 <=j < i <= n) conté els elements l(i,j), 1 <=j < i <= n, de
          la matriu L i la diagonal de A (a(i,i), 1 <= i <= n) conté els elements
          d(i,i), 1 <= i <= n, de de la matriu D.
    """
    n = A.shape[0]
    assert A.shape[1] == n

    #A[1:][0] /= A[0][0]
    for i in range(2, n + 1):
        A[i - 1][0] = A[i - 1][0] / A[0][0]
    for k in range(2, n + 1):
        A[k - 1][k - 1] -= np.sum([
                A[k - 1][r - 1] * A[k - 1][r - 1] * A[r - 1][r - 1]
                for r in range(1, k)
        ])

        for i in range(k + 1, n + 1):
            A[i - 1][k - 1] -= np.sum([
                A[i - 1][r - 1] * A[r - 1][r - 1] * A[k - 1][r - 1]
                for r in range(1, k)
            ])
            A[i - 1][k - 1] /= A[k - 1][k - 1]
    return A


def choleski(A):
    """
    choleski(A): Factorització de Choleski d'una matriu quadrada A.

    Input:
        A: Matriu quadrada d'ordre n x n, que es modificarà in-place per
          contenir les components de L.
    Output:
        A: modificada de manera que la part triangular inferior de A (a(i,j),
          1 <= j < i <= n) conté els elements l(i,j), 1 <= j < i <= n, de la
        matriu L i la diagonal de A (a(i,i), 1 <= i <= n) conté els elements
        l(i,i), 1 <= i <= n, de la matriu L.
    """

    return A


# Solució a partir de la factorització LU de la matriu A del sistema lineal
# Ax = b


def sol_lu(A, b, l_amb_uns_a_la_diagonal=True):
    """
    sol_lu(A, b): Resolució del sistema a partir de la factorització LU de
                  la matriu $A$ (vegeu la descripció dels paràmetres a sota).

    Resol el sistema lineal Ax = b donada la factorització LU de la matriu A que
    proporciona la funció lu(A) (o lu_pp(A), o lu_ppe(A), o doolittle(A), o
    crout(A), o ldlt(A), o choleski(A)). El vector b es modificarà in-place per
    contenir, a la sortida, la solució del sistema lineal Ax = b.

    Remarca: Si A és la matriu que torna la funció lu_pp(A), quan fem la
    factorització LU amb pivot parcial (o la funció lu_ppe(A), quan fem la
    factorització LU amb pivot parcial esglaonat), llavors el vector b ha de ser
    reordenat segons el vector de permutació p que torna la funció lu_pp(A) (o
    lu_ppe(A)) abans de cridar a sol_lu(A, b). En Python, això es pot fer
    simplement fent b = b[p[:]] abans de cridar a sol_lu(A, b).

    Input:
        A: Matriu quadrada d'ordre n x n, tal com surt de la funció lu(A) (o
          lu_pp(A), o lu_ppe(A) o doolittle(A), o crout(A), o ldlt(A), o
          choleski(A):

          Si el paràmetre (opcional) l_amb_uns_a_la_diagonal = True (valor per
          defecte), es suposa que els elements de la diagonal de L són 1s, de
          manera que:

          * La part triangular estrictament inferior de A,
                    a(i,j), 1 <= j < i <= n,
            conté els elements
                    m(i,j), 1 <= j < i <= n,
            de la matriu L, mentre que
          * la part triangular superior de A,
                    a(i,j), 1 <= i <= j <= n,
            conté els elements
                    u(i,j), 1 <= i <= j <= n,
            de la matriu U.

          Per contra, si el paràmetre (opcional) l_amb_uns_a_la_diagonal =
          False, es suposa que els elements de la diagonal de U són 1s, de
          manera que:

          * la part triangular estrictament superior de A,
                    a(i,j), 1 <=i < j <= n,
            conté els elements
                    u(i,j), 1 <=i < j <= n,
            de la matriu U, mentre que

          * la part triangular inferior de A,
                    a(i,j), 1 <= j <= i <= n,
            conté els elements
                    m(i,j), 1 <= j <= i <= n,
            de la matriu L.

        b: Vector (matriu n x 1) qeu conté el terme independent del sistema
          lineal Ax = b.

    Output:
        b: Vector (matriu n x 1) que ara conté la solució del sistema lineal
          Ax = b.
    """
    
    n = b.shape[0]
    for i in range(1, n):
        for j in range(i):
            b[i] -= A[i][j] * b[j]

    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            b[i] -= A[i][j] * b[j]
        b[i] /= A[i][i]

    return b


# Factorització QR: Mètodes de Gram-Schmidt classic, Gram-Schmidt modificat i
# Householder

# TODO modificar a in-place
def gs_classic(a):
    """
    gs_classic(a): Factorització QR d'una matriu a mitjançant el mètode de
                    Gram-Schmidt clàssic.

    Input:
        a: Matriu m x n que es modificarà in-place per contenir les
          components de Q.

    Output:
        a: modificada de manera que ara conté les components de Q
        r: Matriu n x n que conté les components de R.
    """
    n, m = a.shape
    q = np.zeros((m, n))
    r = np.zeros((n, n))
    r[0][0] = np.linalg.norm(a[:, 0])
    print('r', r[0][0])
    q[:, 0] = a[:, 0] / r[0][0]

    for k in range(1, n):
        for j in range(k - 1):
            r[j][k] = np.dot(q[:, j], a[:, k])

        a[:, k] = a[:, k] - sum([r[v][k] * q[:, v] for v in range(k)])
        r[k][k] = np.linalg.norm(a[:, k])
        q[:, k] = a[:, k] / r[k][k]

    return q, r

# TODO modificar a in-place
def gs_modificat(a):
    """
    qr_modificat(a): Factorització QR d'una matriu a mitjançant el mètode
                    modificat de Gram-Schmidt.

    Input:
        a: Matriu m x n que es modificarà in-place per contenir les
          components de Q.

    Output:
        a: modificada de manera que ara conté les components de Q

        r: Matriu n x n que conté les components de R.
    """
    n, m = a.shape

    r = np.zeros((n, n))
    q = np.zeros((n, n))

    r[0][0] = np.linalg.norm(a[:, 0])
    q[:, 0] = a[:, 0] / r[0][0]

    for k in range(1, n):
        for s in range(k, n):
            r[k - 1, s] = np.dot(q[:, k - 1], a[:, s])
            a[:, s] -= r[k - 1, s] * q[:, k - 1]

        r[k][k] = np.linalg.norm(a[:, k])
        q[:, k] = a[:, k] / r[k][k]


    return q, r

def house(x: np.ndarray, tol=1.0e-14) -> Tuple[float, np.ndarray]:
    """
    house(x, tol): Calcula el vector de Householder u i el coeficient beta per a
                   un vector x

    Input:
        x: Vector d'entrada que es vol transformar.

        tol: Tolerància per a la detecció de components petits en x.

    Output:
        beta: Coeficient que determina la matriu de Householder.

        u: Vector de Householder que defineix la matriu de Householder H = I -
           beta * u * u^T.
    """
    x = np.asanyarray(x, dtype=np.float64)
    norm_x = np.linalg.norm(x, ord=2)
    v = - norm_x if x[0] > 0 else norm_x
    u = x.copy()
    u[0] -= v
    if np.abs(v) < tol: beta = 0.0
    else: beta = - 1.0 / (v * u[0])
    return beta, u


def qr_house(A: np.ndarray, tol=1.0e-14):
    """
    qr_house(A): Factorització QR d'una matriu a mitjançant el mètode de
                 Householder.

    Input:
        A: Matriu m x n que conté les components de a.

    Output:
        q: Matriu m x m que conté les components de Q.

        r: Matriu m x n amb les components de R.
    """
    m, n = A.shape

    n_steps = n if m > n else n - 1

    Q = np.eye(m)

    for k in range(n_steps):
        target = A[k:, k]
        if np.max(np.abs(target[1:])) < tol: continue

        beta, u = house(target, tol)
        H = np.eye(m - k) - beta * np.outer(u, u)
        P_k = np.block([
            [np.eye(k),         np.zeros((k, m - k))  ],
            [np.zeros((m - k, k)),  H                   ]
        ])
        Q = Q @ P_k
        A = P_k @ A

    return Q, A
