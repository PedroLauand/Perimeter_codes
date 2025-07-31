import einops
import numpy as np

# Measurements
e_1 = np.array([[-1 - 1j, 0, -2j, -1 + 1j]])
e_2 = np.array([[1 - 1j, 2j, 0, 1 + 1j]])
e_3 = np.array([[-1 + 1j, 2j, 0, -1 - 1j]])
e_4 = np.array([[1 + 1j, 0, -2j, 1 - 1j]])


M_1 = np.matmul(e_1.T, e_1.conj())
M_2 = np.matmul(e_2.T, e_2.conj())
M_3 = np.matmul(e_3.T, e_3.conj())
M_4 = np.matmul(e_4.T, e_4.conj())

M = [M_1.reshape((2,2,2,2)),
     M_2.reshape((2,2,2,2)),
     M_3.reshape((2,2,2,2)),
     M_4.reshape((2,2,2,2))]

# State
psi = np.array([[0, 1, -1, 0]])
rho = np.matmul(psi.T, psi.conj())
rho = rho.reshape((2,2,2,2))


P_ABC = np.zeros((4,4,4), dtype=float)
for a,b,c in np.ndindex((4, 4, 4)):
    P_ABC[a, b, c] = einops.einsum(rho, rho, rho, M[a], M[b], M[c],
                                   (  'ABaket ABbket ABabra ABbbra , '
                                    + 'BCbket BCcket BCbbra BCcbra , '
                                    + 'CAcket CAaket CAcbra CAabra , '
                                    + 'CAaket ABaket CAabra ABabra , '
                                    + 'ABbket BCbket ABbbra BCbbra , '
                                    + 'BCcket CAcket BCcbra CAcbra'
                                    + '-> '))/4096
print(256*P_ABC[0,0,0])
print(256*P_ABC[0,0,1])
print(256*P_ABC[0,1,2])
print(P_ABC.sum())

