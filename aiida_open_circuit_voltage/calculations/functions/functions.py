# -*- coding: utf-8 -*-

from aiida import orm
from aiida.engine import calcfunction
import numpy as np


# not a calcfunction, it's used by workchain to make low and high SOC structures
def get_unique_cation_sites(structure, cations=['Li', 'Mg']):
    '''
    Returns the indices of unique cationic positions in the structure 
    '''
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    structure_pym = structure.get_pymatgen_structure()
    structure_ase = structure.get_ase()
    cation_sites = [site for site in structure_pym.sites if site.species_string in cations]
    symmops = SpacegroupAnalyzer(structure_pym, symprec=1e-5).get_space_group_operations()

    # first site is always unique
    unique_sites = [cation_sites[0], ]
    for site_1 in cation_sites[1:]:
        unique = True
        for site_2 in unique_sites: 
            if symmops.are_symmetrically_equivalent([site_1,], [site_2,], symm_prec=1e-5): unique = False 
        if unique: unique_sites.append(site_1)
    
    all_cation_indices = orm.List(list=[atom.index for atom in structure_ase if atom.symbol in cations])
    unique_cation_indices = orm.List(list=[atom.index for atom in structure_ase for unique_site in unique_sites if (np.around(atom.position, 5) == np.around(unique_site.coords, 5)).all()])

    return all_cation_indices, unique_cation_indices

# @calcfunction
def make_supercell(structure, distance):
    from supercellor import supercell as sc
    pym_sc_struct = sc.make_supercell(structure.get_pymatgen_structure(), distance, verbosity=0, do_niggli_first=False)[0]
    sc_struct = orm.StructureData()
    sc_struct.set_extra('original_unitcell', structure.uuid)
    sc_struct.set_extra('structure_type', 'discharged')
    sc_struct.set_pymatgen(pym_sc_struct)
    return sc_struct

## Using pymatgen symmetric site finder (much faster)
# @calcfunction
def get_low_SOC(structure, unique_indices):
    '''
    Returns a list of structures made after removing 1 symmeterically inquevalent cation.
    We assume that removing one cation from discharged structure doesn't distort the cell.
    '''

    ## Make a list of unique (symmetrically inequivalent) supercells with 1 cation removed
    structure_ase = structure.get_ase()
    low_SOC_structures = []
    for idx in unique_indices:
        low_SOC = structure_ase.copy()
        del low_SOC[idx]
        low_SOC_structures.append(low_SOC)

    ## In case something went wrong with new structure generation
    assert len(low_SOC_structures)==len(unique_indices), f'{len(unique_indices)} unique sites identified by pymatgen, but {len(low_SOC_structures)} unique structures generated'

    ## Store the ase structures as aiida StructureData
    unique_low_SOC_aiida_structures = orm.List(list=[])
    for struct in low_SOC_structures:
        decationised_structure = orm.StructureData()
        decationised_structure.set_extra('original_unitcell', structure.extras['original_unitcell'])
        decationised_structure.set_extra('structure_type', 'low_SOC')
        decationised_structure.set_extra('missing_cations', 1)
        decationised_structure.set_ase(struct)
        decationised_structure.label = decationised_structure.get_formula(mode='count')
        unique_low_SOC_aiida_structures.append(decationised_structure)

    return unique_low_SOC_aiida_structures

## Using ase symmetry comparison
@calcfunction
def get_low_SOC_slow(structure, cations=orm.List(list=['Li', 'Mg'])):
    '''
    Returns a list of structures made after removing 1 symmeterically inquevalent cation.
    We assume that removing one cation from discharged structure doesn't distort the cell.
    '''
    from ase.utils.structure_comparator import SymmetryEquivalenceCheck

    structure_ase = structure.get_ase()
    ## Make a list of all possible supercells with 1 cation removed
    low_SOC_structures = []
    for atom in structure_ase:
        if atom.symbol in cations:
            low_SOC = structure_ase.copy()
            del low_SOC[atom.index]
            low_SOC_structures.append(low_SOC)

    ## Make a list of unique (symmetrically inequivalent) supercells with 1 cation removed
    sym_eq_check = SymmetryEquivalenceCheck(angle_tol=1.0, ltol=0.05, stol=0.1, vol_tol=0.1, scale_volume=False, to_primitive=False)
    unique_low_SOC_structures = [low_SOC_structures[0], ]
    for struct_1 in low_SOC_structures[1:]:
        unique = True
        for struct_2 in unique_low_SOC_structures: 
            if (sym_eq_check.compare(struct_1, struct_2)): unique = False 
        if unique: unique_low_SOC_structures.append(struct_1)

    ## Store the ase structures as aiida StructureData
    unique_low_SOC_aiida_structures = orm.List(list=[])
    for struct in unique_low_SOC_structures:
        decationised_structure = orm.StructureData()
        decationised_structure.set_extra('original_unitcell', structure.extras['original_unitcell'])
        decationised_structure.set_extra('structure_type', 'high_SOC')
        decationised_structure.set_extra('missing_cations', 1)
        decationised_structure.set_ase(struct)
        decationised_structure.label = decationised_structure.get_formula(mode='count')
        unique_low_SOC_aiida_structures.append(decationised_structure)

    return unique_low_SOC_aiida_structures

# @calcfunction
def get_high_SOC(discharged_structure, charged_structure, all_cation_indices, unique_indices):
    '''
    Returns a list of structures made after removing all but 1 symmeterically inquevalent cation
    i.e. a structure that contains only 1 cation.
    We assume that adding one cation to a completely charged structure doesn't distort the cell.
    Since it is easier to remove atoms than add one at specific sites, we scale the discharged 
    supercell wrt the lattice vectors of charged supercell and then remove all but one cation.
    '''

    discharged_ase = discharged_structure.get_ase()
    charged_ase = charged_structure.get_ase()
    ## scaling the discharged supercell
    discharged_ase.set_cell(charged_ase.get_cell(), scale_atoms=True)

    ## Make a list of all possible supercells with only 1 cation remaining
    high_SOC_structures = []
    for idx in unique_indices:
        tmp_indices = all_cation_indices.get_list().copy()
        high_SOC = discharged_ase.copy()
        ## keeping all but one inequivalent cation
        tmp_indices.remove(idx)
        del high_SOC[tmp_indices]
        high_SOC_structures.append(high_SOC)
    ## In case somethign went wrong with new structure generation
    assert len(high_SOC_structures)==len(unique_indices), f'{len(unique_indices)} unique sites identified by pymatgen, but {len(high_SOC_structures)} unique structures generated'

    ## Store the ase structures as aiida StructureData
    unique_high_SOC_aiida_structures = []
    for struct in high_SOC_structures:
        decationised_structure = orm.StructureData()
        decationised_structure.set_extra('original_unitcell', discharged_structure.extras['original_unitcell'])
        decationised_structure.set_extra('structure_type', 'high_SOC')
        decationised_structure.set_extra('missing_cations', len(all_cation_indices)-1)
        decationised_structure.set_ase(struct)
        decationised_structure.label = decationised_structure.get_formula(mode='count')
        unique_high_SOC_aiida_structures.append(decationised_structure)

    return orm.List(list=unique_high_SOC_aiida_structures)

# @calcfunction
def get_charged(structure, cation_to_remove=orm.List(list=['Li', 'Mg'])):
    """
    Take the input structure and build a completely charged structure i.e. 
    structure containing no cations
    """
    
    assert isinstance(structure, orm.StructureData), "input structure needs to be an instance of {}".format(orm.StructureData)

    struct_ase = structure.get_ase()
    cations_indices = [atom.index for atom in struct_ase if atom.symbol in cation_to_remove]
    del struct_ase[cations_indices]

    decationised_structure = orm.StructureData()
    decationised_structure.set_extra('original_unitcell', structure.uuid)
    decationised_structure.set_extra('structure_type', 'charged')
    decationised_structure.set_extra('missing_cations', len(cations_indices))
    decationised_structure.set_ase(struct_ase)
    decationised_structure.label = decationised_structure.get_formula(mode='count')

    return decationised_structure

