from typing import Literal

# Per assegurar que carreguem els fitxers correctes.
Fitxers = Literal["sistema.npz"] | Literal["sistemaedd.npz"] | Literal["sistemasdp.npz"]

# String que es concatena al principi de totes les rutes de Fitxers.
DIRECTORI_DADES = "data/"
