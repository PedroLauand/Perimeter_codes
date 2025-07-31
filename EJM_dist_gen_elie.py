import einops
import numpy as np

# Measurements
e_1 = np.array([[-1 - 1j, 0, -2j, -1 + 1j]]).reshape((2, 2))
e_2 = np.array([[1 - 1j, 2j, 0, 1 + 1j]]).reshape((2, 2))
e_3 = np.array([[-1 + 1j, 2j, 0, -1 - 1j]]).reshape((2, 2))
e_4 = np.array([[1 + 1j, 0, -2j, 1 - 1j]]).reshape((2, 2))
M = np.array([e_1, e_2, e_3, e_4])

# State
psi = np.array([[0, 1, -1, 0]]).reshape((2, 2))
P_ABC = np.zeros((4,4,4), dtype=float)
for a,b,c in np.ndindex((4, 4, 4)):
    amplitude = einops.einsum(psi, psi, psi, M[a], M[b], M[c],
                             'ABa ABb, BCb BCc, CAc CAa, CAa ABa, ABb BCb, BCc CAc-> ')
    P_ABC[a, b, c] = np.real(amplitude*amplitude.conj())
normalization = 2*2*2*8*8*8
P_ABC = P_ABC/normalization

print(256*P_ABC[0,0,0])
print(256*P_ABC[0,0,1])
print(256*P_ABC[0,1,2])
print(P_ABC.sum())

### Now demoing line inflation
P_ABC_line = np.zeros((4,4,4), dtype=float)
for a,b,c in np.ndindex((4, 4, 4)):
    leftover_state = einops.einsum(psi, psi, psi, psi, M[a], M[b], M[c],
                             'JAg JAa, ABa ABb, BCb BCc, JCc JCg, JAa ABa, ABb BCb, BCc JCc -> JAg JCg')

    P_ABC_line[a, b, c] = np.real(np.dot(leftover_state.ravel(), leftover_state.ravel().conj()))
normalization = 8*8*8*2*2*2*2
P_ABC_line = P_ABC_line/normalization

print(256*P_ABC_line[0,0,0])
print(256*P_ABC_line[0,0,1])
print(256*P_ABC_line[0,1,2])
print(P_ABC_line.sum())
