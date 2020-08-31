import scm.plams
from scm import plams
from scm.plams import *
import numpy as np
import os
from os import listdir
import pandas as pd
import sys
import copy
from project_utilities import Mol

def extract_overlap(results,save,folder,save_as,method,logger = False):
    kfpath = results.rkfpath()
    kfpath = kfpath[:-7] + 'dftb.rkf'
    mykf = KFFile(kfpath)
    S = mykf.read('Matrices','Data(1)')
    H = mykf.read('Matrices','Data(2)')
    energy = mykf.read('AMSResults','Energy')
    orbitals = mykf.read('Orbitals','Energies(1)')
    coefs = mykf.read('Orbitals','Coefficients(1)')
    coefs = np.array(coefs).reshape(int(np.sqrt(len(coefs))),int(np.sqrt(len(coefs))))
    charges = np.array(mykf.read('AMSResults','Charges'))
    F = np.array(H).reshape(int(np.sqrt(len(H))),int(np.sqrt(len(H))))
    S = np.array(S).reshape(int(np.sqrt(len(S))),int(np.sqrt(len(S))))
    orbitals = np.array(orbitals)
    energies = {}
    energies['Total'] = mykf.read('Properties','Value(1)')
    energies['Electronic'] = mykf.read('Properties','Value(2)')
    energies['Coulomb'] = mykf.read('Properties','Value(3)')
    energies['Repulsion'] = mykf.read('Properties','Value(4)')
    occupations = mykf.read('Orbitals','Occupations(1)')
    if save:
        if folder == None:
            foldername  = 'DFTB_results'
        else:
            foldername = folder
        full = foldername + '/' + save_as
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        full_cache = full
        counter = 1
        while os.path.exists(full_cache):
            full_cache = full + '_' + str(counter)
            counter += 1
        os.makedirs(full_cache)
        full = full_cache + '/'
        np.savetxt(full + 'F',F)
        np.savetxt(full + 'S',S)
        np.savetxt(full + 'energies',orbitals)
        np.savetxt(full + 'coefs',coefs)
        np.savetxt(full + 'energy',np.array([energy]))
        np.savetxt(full + 'charges',charges)
        f = open(full + 'name','w')
        f.write(save_as)
        f.close()
        print('Output saved to: ' + full)
    if not logger:
        if method != 'SCC-DFTB':
            return F,S,orbitals,coefs,energies,charges, occupations
        else:
            gamma = mykf.read('Matrices','Data(3)')
            gamma = np.array(gamma).reshape(int(np.sqrt(len(gamma))),int(np.sqrt(len(gamma))))
            return  F,S,gamma,orbitals,coefs,energies,charges, occupations
    if logger:
        entries = mykf.read('SCCLogger','nEntries')
        errors = []
        errors1 = []
        gamma = mykf.read('Matrices','Data(3)')
        gamma = np.array(gamma).reshape(int(np.sqrt(len(gamma))),int(np.sqrt(len(gamma))))
        electronic = mykf.read('Properties','Value(2)')
        coulomb = mykf.read('Properties','Value(3)')
        for i in range(1,entries + 1):
            errors.append(mykf.read('SCCLogger','dqMax(' + str(i)+')'))
            errors1.append(mykf.read('SCCLogger','Eout(' + str(i)+')'))
        return np.array([errors,errors1]),gamma, electronic + coulomb

def DFTB(mol, parameter_set = 'DFTB.org/3ob-3-1', erase_dir = False, name = None, folder = None, save = True, method = 'DFTB',logger = False):
    config.erase_workdir = erase_dir
    s = Settings()
    #AMS driver input
    s.input.ams.Task = 'SinglePoint'
    #s.input.ams.GeometryOptimization.Convergence.Gradients = 1.0e-4
    #s.input.ams.Properties.NormalModes = 'true'
    s.input.DFTB.StoreMatrices = 'yes'
    #DFTB engine input
    s.input.DFTB.Model = method
    s.input.DFTB.ResourcesDir = parameter_set
    param_name = parameter_set.replace('/','_')

    if type(mol) == str:
        path_to_xyz = mol
        if "/" in path_to_xyz:
            for i in range(len(path_to_xyz)):
                if path_to_xyz[i] == '/':
                    filename = path_to_xyz[i+1:]
        else:
            filename = path_to_xyz
        if name == None:
            if filename[-4:] == '.xyz':
                j = AMSJob(molecule=plams_mol, settings=s, name = filename[:-4] + '_' + param_name)
                save_as = filename[:-4] + '_' + param_name
            else:
                j = AMSJob(molecule=plams_mol, settings=s, name = filename + '_' + param_name)
                save_as = filename + '_' + param_name
        else:
            j = AMSJob(molecule=plams_mol, settings=s, name = name + '_' + param_name)
            save_as = name + '_' + param_name
        results = j.run()

    else:
        path_to_xyz = mol.xyz_path
        if type(mol.name) == str:
            filename = mol.name
        elif type(path_to_xyz) == str:
            if "/" in path_to_xyz:
                for i in range(len(path_to_xyz)):
                    if path_to_xyz[i] == '/':
                        filename = path_to_xyz[i+1:]
            else:
                filename = path_to_xyz
        else:
            filename = 'unnamed_mol'
        plams_mol = Molecule()
        for i,k in zip(mol.atom_numbers(),mol.coords):
            plams_mol.add_atom(Atom(atnum = i,coords = (k[0],k[1],k[2])))
        if name == None:
            if len(filename) >= 4:
                if filename[-4:] == '.xyz':
                    j = AMSJob(molecule=plams_mol, settings=s, name = filename[:-4] + '_' + param_name)
                    save_as = filename[:-4] + '_' + param_name
                else:
                    j = AMSJob(molecule=plams_mol, settings=s, name = filename + '_' + param_name)
                    save_as = filename + '_' + param_name
            else:
                j = AMSJob(molecule=plams_mol, settings=s, name = filename + '_' + param_name)
                save_as = filename + '_' + param_name
        else:
            j = AMSJob(molecule=plams_mol, settings=s, name = name + '_' + param_name)
            save_as = name + '_' + param_name
        results = j.run()
    return extract_overlap(results,save,folder,save_as,method,logger = logger)

def SCC_DFTB(mol, parameter_set = 'DFTB.org/3ob-3-1', erase_dir = False, name = None, folder = 'SCC-DFTB_results', save = True, method = 'SCC-DFTB',logger = False):
    return DFTB(mol = mol, parameter_set = parameter_set, erase_dir = erase_dir, name = name, folder = folder, save = save, method = method,logger = logger)
