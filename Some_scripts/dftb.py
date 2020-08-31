from project_utilities import *
import numba
import numpy as np

@numba.njit(parallel = True)
def electronic_energy_numba(occs,vecs,H0):
    E = 0
    for a in range(len(occs)):
        for u in range(len(vecs)):
            for v in range(len(vecs)):
                E += occs[a]*vecs[u,a]*vecs[v,a]*H0[u,v]
    return E

def electronic_energy(occs,C,H0):
    return electronic_energy_numba(occs,C,H0)

def coulombE(mol,gamma,charges):
    atoms = mol.atom_numbers()
    E = 0
    for i in range(len(atoms)):
        for j in range(len(atoms)):
            E+=charges[i]*gamma[i,j]*charges[j]
    return (1/2)*E

@numba.njit(parallel = True)
def numba_charce_calc(atoms,coeffs,occupations,S,mapping,num_H = 1,num_O = 4,num_S = 9):
    charges = np.zeros(len(atoms),dtype = np.float64)
    C = coeffs.copy()
    electrons = np.array([0,1,2,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8])
    num_elec = np.zeros(len(atoms))
    number_of_pairs = 0
    for i in occupations:
        number_of_pairs += i
    number_of_pairs /= 2
    number_of_pairs = int(number_of_pairs)
    for i in range(len(atoms)):
        index = atoms[i]
        num_elec[i] = electrons[index]
    for j in range(len(atoms)):
        offset = 0
        for i in range(j):
            offset += mapping[atoms[i]]
        ranging = np.array([offset,offset + mapping[atoms[j]]])
        for i in range(number_of_pairs):
            for u in range(ranging[0],ranging[1]):
                for v in range(S.shape[0]):
                    a = occupations[i]*(C[u,i]*S[u,v]*C[v,i] + C[v,i]*S[v,u]*C[u,i])
                    charges[j] += a
    return num_elec - (1/2)*charges

def charge_calc(mol,coeffs,occupations,S,num_H = 1,num_O = 4,num_S = 9):
    mapping = [0,num_H,num_H,num_O,num_O,num_O,num_O,num_O,num_O,num_O,num_O,num_S,num_S,num_S,num_S,num_S,num_S,num_S,num_S]
    mapping = np.array([int(x) for x in mapping])
    atoms = mol.atom_numbers()
    atoms = np.array([int(x) for x in atoms])
    occupations = np.array([int(x) for x in occupations])
    return numba_charce_calc(atoms,coeffs,occupations,S,mapping,num_H = num_H,num_O = num_O,num_S = num_S)

@numba.njit(parallel = True)
def numba_hamiltonian(atoms,charges,gamma,mapping,S):
    corrected = np.zeros(gamma.shape)
    for i in range(gamma.shape[0]):
        for j in range(gamma.shape[1]):
            for a in range(charges.shape[0]):
                corrected[i,j] += (gamma[i,a] + gamma[j,a])*charges[a]
    corrected2 = np.zeros(S.shape)
    for j in range(len(atoms)):
        offset = 0
        for i in range(j):
            offset += mapping[atoms[i]]
        ranging = np.array([offset,offset + mapping[atoms[j]]])
        for i in range(len(atoms)):
            if i >= j:
                offset = 0
                for k in range(i):
                    offset += mapping[atoms[k]]
                ranging2 = np.array([offset,offset + mapping[atoms[i]]])
                corrected2[ranging[0]:ranging[1],ranging2[0]:ranging2[1]] = corrected[i,j]
                corrected2[ranging2[0]:ranging2[1],ranging[0]:ranging[1]] = corrected[j,i]
    return corrected2

def hamiltonian(mol,H0,S,charges,gamma,num_H = 1,num_O = 4,num_S = 9):
    atoms = mol.atom_numbers()
    atoms = np.array([int(x) for x in atoms])
    mapping = [0,num_H,num_H,num_O,num_O,num_O,num_O,num_O,num_O,num_O,num_O,num_S,num_S,num_S,num_S,num_S,num_S,num_S,num_S]
    mapping = np.array([int(x) for x in mapping])
    corrected = numba_hamiltonian(atoms,charges,gamma,mapping,S)
    corrected = H0 - (1/2)*S*corrected
    return corrected

def SCC(mol,occupations,H0,S,gamma,num_H = 1,num_O = 4,num_S = 9,iterations = 10,charge = [None]):
    charges = np.zeros((iterations,len(mol.atoms)))
    energies = np.zeros(iterations)
    H = H0.copy()
    if charge[0] != None:
        H = hamiltonian(mol,H0,S,charge,gamma,num_H = num_H,num_O = num_O,num_S = num_S)
    for i in range(iterations):
        eigs,vecs = eig(np.linalg.inv(S)@ H)
        SB = vecs.T @ S @ vecs
        norm_vecs = normalize(vecs,SB)
        charge = charge_calc(mol,norm_vecs,occupations,S,num_H = num_H,num_O = num_O,num_S = num_S)
        charges[i] = charge
        energies[i] = coulombE(mol,gamma,charge)
        H = hamiltonian(mol,H0,S,charge,gamma,num_H = num_H,num_O = num_O,num_S = num_S)
    return charges,energies
