import numpy as np

def invertir_diagonal(D: np.ndarray):
    elements = np.diag(D).copy()

    for i in range(len(elements)):
        if elements[i] != 0:
            elements[i] = 1 / elements[i]

    return np.diag(elements)

# A les següents funcions, he eliminat el paràmetre uns_a_la_diagonal perquè
# si la matriu ho compleix, aleshores simplement les divisions no tindran efecte.
# Això simplifica la crida a sol_sti i sol_sts en les funcions de LUGaussiana.


def sol_sti(A: np.ndarray, b: np.ndarray):
    """
    sol_sti(A, b): Resolució de un sistema triangular inferior amb 1s a la
                  diagonal.

    Resol el sistema lineal Ax = b, on A és una matriu triangular amb 1s a la
    diagonal.

    El vector b es modificarà in-place per contenir, a la sortida, la solució
    del sistema lineal Ax = b.

    Input:
        A: Matriu triangular inferior d'ordre n x n.

        b: Matriu n x 1 que conté el terme independent del sistema.

    Output:
        b: Matriu n x 1 que ara conté la solució del sistema.
    """
    n = A.shape[0]
    assert A.shape[1] == n
    for i in range(n):
        for j in range(i):
            b[i] -= A[i][j] * b[j]
        b[i] /= A[i][i]
    return b


def sol_sts(A: np.ndarray, b: np.ndarray):
    """
    sol_sts(A,b): Resolució d'un sistema triangular superior.

    Resol el sistema lineal Ax = b, on A és una matriu triangular superior
    no singular.

    El vector b es modificarà in-place per contenir, a la sortida, la solució
    del sistema lineal Ax = b.

    Input:
        A: Matriu triangular superior d'ordre n x n.

        b: Matriu n x 1 que conté el terme independent del sistema.

    Output:
        b: Matriu n x 1 que ara conté la solució del sistema.
    """
    n = A.shape[0]
    assert A.shape[1] == n
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            b[i] -= A[i][j] * b[j]
        b[i] /= A[i][i]
    return b

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
    """
    eigvals = np.linalg.eigvals(A)
    return np.max(np.abs(eigvals))