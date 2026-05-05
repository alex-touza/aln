import numpy as np

def invertir_diagonal(D: np.ndarray):
    elements = np.diag(D).copy()

    for i in range(len(elements)):
        if elements[i] != 0:
            elements[i] = 1 / elements[i]

    return np.diag(elements)


def sol_sti(A: np.ndarray, b: np.ndarray, uns_a_la_diagonal=True):
    """
    sol_sti(A, b): Resolució de un sistema triangular inferior amb 1s a la
                  diagonal.

    Resol el sistema lineal Ax = b, on A és una matriu triangular amb 1s a la
    diagonal.

    El vector b es modificarà in-place per contenir, a la sortida, la solució
    del sistema lineal Ax = b.

    Input:
        A: Matriu triangular inferior d'ordre n x n, amb 1s a la diagonal si el
          paràmetre (opcional) uns_a_la_diagonal = True.

        b: Matriu n x 1 que conté el terme independent del sistema.

        uns_a_la_diagonal: (opcional) si és True (valor per defecte), la matriu
          A té 1s a la diagonal.

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


def sol_sts(A: np.ndarray, b: np.ndarray, uns_a_la_diagonal=False):
    """
    sol_sts(A,b): Resolució d'un sistema triangular superior.

    Resol el sistema lineal Ax = b, on A és una matriu triangular superior
    no singular.

    El vector b es modificarà in-place per contenir, a la sortida, la solució
    del sistema lineal Ax = b.

    Input:
        A: Matriu triangular superior d'ordre n x n, amb 1s a la diagonal si el
          paràmetre (opcional) uns_a_la_diagonal = True.

        b: Matriu n x 1 que conté el terme independent del sistema.

        uns_a_la_diagonal: (opcional) si és True, la matriu A té 1s a la
           digonal.

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
