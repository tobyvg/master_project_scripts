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

atom_symbols = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

def extract(results,save_as, save_geom,find_symm,folder):
    kfpath = results._kfpath()
    mykf = KFFile(kfpath)
    if find_symm:
        symm = mykf.read('Geometry','Geometric Symmetry')
        symm = symm.replace(' ','')
        save_as += '_' + symm
    new_mol = results.get_main_molecule()
    coords = np.array([i.coords for i in new_mol])
    atnumbers = np.array([i.atnum for i in new_mol])
    atoms = np.array([atom_symbols[i-1] for i in atnumbers])
    if save_geom:
        if folder == None:
            foldername  = 'DFT_geomopt'
        else:
            foldername = folder
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        counter = 0
        cache = foldername + '/' + save_as +'.xyz'
        while os.path.exists(cache):
            cache = foldername + '/' + save_as + '_' + str(counter+1) + '.xyz'
            counter = counter + 1
        towrite = ''
        towrite += str(len(coords)) + '\n'
        towrite += save_as
        for a,cs in zip(atoms,coords):
            towrite += '\n' + a
            for c in cs:
                towrite += '  ' + str(c)
        f = open(cache,'w')
        f.write(towrite)
        f.close()
        print('Geometry saved as: '+ cache)
    return coords,atnumbers ##coordinates, atom_numbers


def DFT_geomopt(mol, basis = 'TZP',functional_GGA = 'PBE', erase_dir = False, name = None, save_geom = True,find_symm = False,folder = None):
    s = Settings()
    #AMS driver input
    s.task = 'Geometry Optimization'
    s.input.geometry
    #s.input.ams.GeometryOptimization.Convergence.Gradients = 1.0e-4
    #s.input.ams.Properties.NormalModes = 'true'
    s.input.BASIS.type = basis
    s.input.BASIS.core = 'None'
    s.input.XC.GGA = functional_GGA
    config.erase_workdir = erase_dir
    if type(mol) == str:
        path_to_xyz = mol
        if "/" in path_to_xyz:
            for i in range(len(path_to_xyz)):
                if path_to_xyz[i] == '/':
                    filename = path_to_xyz[i+1:]
        else:
            filename = path_to_xyz
        plams_mol = Molecule(path_to_xyz)


        #DFTB engine input
        if name == None:
            if filename[-4:] == '.xyz':
                j = ADFJob(molecule=plams_mol, settings=s, name = filename[:-4] + '_' + functional_GGA  + basis + '_geomopt')
                save_as = filename[:-4] + '_' + functional_GGA  + basis + '_geomopt'
            else:
                j = ADFJob(molecule=plams_mol, settings=s, name = filename + '_' + functional_GGA  + basis+ '_geomopt')
                save_as = filename + '_' + functional_GGA  + basis+ '_geomopt'
        else:
            j = ADFJob(molecule=plams_mol, settings=s, name = name + '_' + functional_GGA  + basis+ '_geomopt')
            save_as = name + '_' + functional_GGA  + basis + '_geomopt'
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
                    j = ADFJob(molecule=plams_mol, settings=s, name = filename[:-4] + '_' + functional_GGA + '_' + basis+ '_geomopt')
                    save_as = filename[:-4] + '_' + functional_GGA + '_' + basis+ '_geomopt'
                else:
                    j = ADFJob(molecule=plams_mol, settings=s, name = filename + '_' + functional_GGA +'_' + basis+ '_geomopt')
                    save_as = filename + '_' + functional_GGA +'_' + basis+ '_geomopt'
            else:
                j = ADFJob(molecule=plams_mol, settings=s, name = filename + '_' + functional_GGA + '_' + basis+ '_geomopt')
                save_as = filename + '_' + functional_GGA + '_' + basis+ 'geomopt'
        else:
            j = ADFJob(molecule=plams_mol, settings=s, name = name + '_' + functional_GGA + '_' + basis+ '_geomopt')
            save_as = name + '_' + functional_GGA + '_' + basis+ '_geomopt'
        results = j.run()
    return extract(results, save_as, save_geom,find_symm,folder)
