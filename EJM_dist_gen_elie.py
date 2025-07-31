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
joint_state = einops.einsum(psi, psi, psi,
                             'Al Ar, Bl Br, Cl Cr -> Al Ar Bl Br Cl Cr')
P_ABC = np.zeros((4,4,4), dtype=float)
for a,b,c in np.ndindex((4, 4, 4)):
    amplitude = einops.einsum(joint_state, M[a], M[b], M[c],
                             'Al Ar Bl Br Cl Cr, Ar Bl, Br Cl, Cr Al -> ')
    P_ABC[a, b, c] = np.real(amplitude*amplitude.conj())/4096

print(256*P_ABC[0,0,0])
print(256*P_ABC[0,0,1])
print(256*P_ABC[0,1,2])
print(P_ABC.sum())
